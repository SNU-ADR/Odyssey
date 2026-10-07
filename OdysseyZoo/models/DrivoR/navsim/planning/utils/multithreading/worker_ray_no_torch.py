import contextlib
import logging
import os
import tempfile
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Union

import ray
from psutil import cpu_count

from nuplan.planning.utils.multithreading.ray_execution import ray_map
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool, WorkerResources

logger = logging.getLogger(__name__)

# Silent botocore which is polluting the terminal because of serialization and deserialization
# with following message: INFO:botocore.credentials:Credentials found in config file: ~/.aws/config
logging.getLogger("botocore").setLevel(logging.WARNING)


# Local Ray heads on one node probe for unused ports independently of each other, so two jobs
# starting at the same instant can pick the same port. Serialize head startup per node. Within a
# DDP job only local rank 0 starts a head; the other ranks attach to its explicit GCS address.
_RAY_INIT_ATTEMPTS = 3
_RAY_INIT_LOCK_TIMEOUT_S = 300
_RANK0_MARKER_TIMEOUT_S = 300


def _ray_proc_suffix() -> str:
    """
    :return: a token that is unique among the processes of one distributed job on this node.
    """
    for env_var in ("LOCAL_RANK", "RANK", "NODE_RANK", "SLURM_PROCID"):
        value = os.environ.get(env_var, "")
        if value.isdigit():
            return f"r{value}"
    return f"p{os.getpid()}"


def _short_ray_temp_dir() -> str:
    """
    Ray puts its plasma/raylet AF_UNIX sockets under <temp_dir>/session_<stamp>_<pid>/sockets/,
    and an AF_UNIX path cannot exceed 107 bytes. A long TMPDIR or RAY_TMPDIR (e.g. one on a
    shared filesystem, set to avoid a noexec /tmp) makes ray.init() die with
    "AF_UNIX path length cannot exceed 107 bytes" before any raylet starts.

    One local Ray head is shared by all DDP ranks. Give that head a private directory next to
    RAY_TMPDIR (or the system temp dir), falling back to a short one under /tmp when that is too
    long; non-zero ranks attach to its explicit address and do not consult Ray's session_latest
    bookkeeping.
    :return: a short node-local temp dir private to the rank-0 Ray head.
    """
    # Ray's temp root: $RAY_TMPDIR, else <system temp dir>/ray
    default_dir = os.environ.get("RAY_TMPDIR") or os.path.join(tempfile.gettempdir(), "ray")
    temp_dir = f"{default_dir}_{_ray_proc_suffix()}"
    # "/session_<19 char stamp>_<pid>/sockets/plasma_store" is ~61 bytes; keep headroom.
    if len(temp_dir) + 64 > 107:
        temp_dir = os.path.join("/tmp", f"ray_{os.getuid()}_{_ray_proc_suffix()}")
        logger.info(f"Ray temp dir {default_dir} is too long for AF_UNIX sockets -> using {temp_dir}")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def _rank0_marker_path() -> Path:
    """Return the rendezvous file shared by Lightning's rank-0 parent and its DDP children."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    # In Lightning's subprocess DDP strategy, ranks 1..N are direct children of the original
    # process, which remains rank 0. Encoding that parent PID keeps concurrent jobs independent.
    rank0_pid = os.getppid() if local_rank != 0 else os.getpid()
    marker_dir = Path("/tmp/ray")
    marker_dir.mkdir(parents=True, exist_ok=True)
    return marker_dir / f"drivor_rank0_{rank0_pid}.ready"


def _wait_for_rank0_address(marker_path: Path) -> str:
    """Wait until rank 0 publishes a live PID and its explicit Ray GCS address."""
    deadline = time.time() + _RANK0_MARKER_TIMEOUT_S
    while True:
        try:
            lines = marker_path.read_text().splitlines()
            marker_pid = int(lines[0])
            address = lines[1]
        except (FileNotFoundError, IndexError, ValueError):
            marker_pid, address = None, ""
        if marker_pid is not None and address and os.path.exists(f"/proc/{marker_pid}"):
            return address
        if time.time() > deadline:
            raise RuntimeError(
                "Timed out waiting for rank 0 to start the local Ray cluster "
                f"(expected a live-PID marker at {marker_path})."
            )
        time.sleep(1)


@contextlib.contextmanager
def _ray_init_gate() -> Iterator[None]:
    """
    Hold a node-local lock so only one local ray head starts at a time (see the comment above
    _RAY_INIT_ATTEMPTS).
    Falls through without the lock if it cannot be taken, so a stale lock never blocks training.
    """
    try:
        from filelock import FileLock, Timeout
    except ImportError:
        yield
        return
    lock = FileLock(os.path.join("/tmp", f"ray_init_{os.getuid()}.lock"), timeout=_RAY_INIT_LOCK_TIMEOUT_S)
    try:
        lock.acquire()
    except Timeout:
        logger.warning(f"ray init lock busy for {_RAY_INIT_LOCK_TIMEOUT_S}s -> starting ray without it")
        yield
        return
    try:
        yield
    finally:
        lock.release()


def initialize_ray(
    master_node_ip: Optional[str] = None,
    threads_per_node: Optional[int] = None,
    local_mode: bool = False,
    log_to_driver: bool = True,
    use_distributed: bool = False,
) -> WorkerResources:
    """
    Initialize ray worker.
    ENV_VAR_MASTER_NODE_IP="master node IP".
    ENV_VAR_MASTER_NODE_PASSWORD="password to the master node".
    ENV_VAR_NUM_NODES="number of nodes available".
    :param master_node_ip: if available, ray will connect to remote cluster.
    :param threads_per_node: Number of threads to use per node.
    :param log_to_driver: If true, the output from all of the worker
            processes on all nodes will be directed to the driver.
    :param local_mode: If true, the code will be executed serially. This
            is useful for debugging.
    :param use_distributed: If true, and the env vars are available,
            ray will launch in distributed mode
    :return: created WorkerResources.
    """
    # Env variables which are set through SLURM script
    env_var_master_node_ip = "ip_head"
    env_var_master_node_password = "redis_password"
    env_var_num_nodes = "num_nodes"

    # Read number of CPU cores on current machine
    number_of_cpus_per_node = threads_per_node if threads_per_node else cpu_count(logical=True)
    number_of_gpus_per_node = 0  # no cuda support
    if not number_of_gpus_per_node:
        logger.info("Not using GPU in ray")

    # Find a way in how the ray should be initialized
    if master_node_ip and use_distributed:
        # Connect to ray remotely to node ip
        logger.info(f"Connecting to cluster at: {master_node_ip}!")
        ray.init(address=f"ray://{master_node_ip}:10001", local_mode=local_mode, log_to_driver=log_to_driver)
        number_of_nodes = 1
    elif env_var_master_node_ip in os.environ and use_distributed:
        # In this way, we started ray on the current machine which generated password and master node ip:
        # It was started with "ray start --head"
        number_of_nodes = int(os.environ[env_var_num_nodes])
        master_node_ip = os.environ[env_var_master_node_ip].split(":")[0]
        redis_password = os.environ[env_var_master_node_password].split(":")[0]
        logger.info(f"Connecting as part of a cluster at: {master_node_ip} with password: {redis_password}!")
        # Connect to cluster, follow to https://docs.ray.io/en/latest/package-ref.html for more info
        ray.init(
            address="auto",
            _node_ip_address=master_node_ip,
            _redis_password=redis_password,
            log_to_driver=log_to_driver,
            local_mode=local_mode,
        )
    elif int(os.environ.get("LOCAL_RANK", "0")) != 0:
        number_of_nodes = 1
        marker_path = _rank0_marker_path()
        logger.info(f"Non-zero LOCAL_RANK: waiting for rank 0's Ray cluster at {marker_path}.")
        ray.init(
            address=_wait_for_rank0_address(marker_path),
            local_mode=local_mode,
            log_to_driver=log_to_driver,
        )
    else:
        # In this case, we will just start ray directly from this script
        number_of_nodes = 1
        logger.info("Starting ray local!")
        temp_dir = _short_ray_temp_dir()
        marker_path = _rank0_marker_path()
        marker_path.unlink(missing_ok=True)
        for attempt in range(1, _RAY_INIT_ATTEMPTS + 1):
            try:
                with _ray_init_gate():
                    context = ray.init(
                        num_cpus=number_of_cpus_per_node,
                        dashboard_host="0.0.0.0",
                        local_mode=local_mode,
                        log_to_driver=log_to_driver,
                        _temp_dir=temp_dir,
                    )
                break
            except Exception as exc:  # a lost port race kills the whole DDP job, so retry it
                if attempt == _RAY_INIT_ATTEMPTS:
                    raise
                logger.warning(f"ray.init failed ({exc}) -> retry {attempt}/{_RAY_INIT_ATTEMPTS - 1}")
                with contextlib.suppress(Exception):
                    ray.shutdown()
                time.sleep(5 * attempt)
        address = context.address_info.get("gcs_address") or context.address_info.get("address")
        if not address:
            raise RuntimeError("ray.init did not return a GCS address for DDP rank rendezvous")
        marker_path.write_text(f"{os.getpid()}\n{address}\n")

    return WorkerResources(
        number_of_nodes=number_of_nodes,
        number_of_cpus_per_node=number_of_cpus_per_node,
        number_of_gpus_per_node=number_of_gpus_per_node,
    )


class RayDistributedNoTorch(WorkerPool):
    """
    This worker uses ray to distribute work across all available threads.
    """

    def __init__(
        self,
        master_node_ip: Optional[str] = None,
        threads_per_node: Optional[int] = None,
        debug_mode: bool = False,
        log_to_driver: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        logs_subdir: Optional[str] = "logs",
        use_distributed: bool = False,
    ):
        """
        Initialize ray worker.
        :param master_node_ip: if available, ray will connect to remote cluster.
        :param threads_per_node: Number of threads to use per node.
        :param debug_mode: If true, the code will be executed serially. This
            is useful for debugging.
        :param log_to_driver: If true, the output from all of the worker
                processes on all nodes will be directed to the driver.
        :param output_dir: Experiment output directory.
        :param logs_subdir: Subdirectory inside experiment dir to store worker logs.
        :param use_distributed: Boolean flag to explicitly enable/disable distributed computation
        """
        self._master_node_ip = master_node_ip
        self._threads_per_node = threads_per_node
        self._local_mode = debug_mode
        self._log_to_driver = log_to_driver
        self._log_dir: Optional[Path] = Path(output_dir) / (logs_subdir or "") if output_dir is not None else None
        self._use_distributed = use_distributed
        super().__init__(self.initialize())

    def initialize(self) -> WorkerResources:
        """
        Initialize ray.
        :return: created WorkerResources.
        """
        # In case ray was already running, shut it down. This occurs mainly in tests
        if ray.is_initialized():
            logger.warning("Ray is running, we will shut it down before starting again!")
            ray.shutdown()

        return initialize_ray(
            master_node_ip=self._master_node_ip,
            threads_per_node=self._threads_per_node,
            local_mode=self._local_mode,
            log_to_driver=self._log_to_driver,
            use_distributed=self._use_distributed,
        )

    def shutdown(self) -> None:
        """
        Shutdown the worker and clear memory.
        """
        ray.shutdown()

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool = False) -> List[Any]:
        """Inherited, see superclass."""
        del verbose
        return ray_map(task, *item_lists, log_dir=self._log_dir)  # type: ignore

    def submit(self, task: Task, *args: Any, **kwargs: Any):
        """Inherited, see superclass."""
        remote_fn = ray.remote(task.fn).options(num_gpus=task.num_gpus, num_cpus=task.num_cpus)
        object_ids: ray._raylet.ObjectRef = remote_fn.remote(*args, **kwargs)
        return object_ids.future()  # type: ignore
