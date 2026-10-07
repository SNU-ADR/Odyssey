"""Speak to an out-of-process image restorer over a simple binary wire protocol.

WHY THIS EXISTS. Requirement: the images the planner sees must be restored, and
the restorer must be swappable without touching the render path again.

The render path has exactly one restoration hook: mtgs.py calls
`restorer_hook.get_restorer()` and, if it returns something, hands it the undistorted
camera batch immediately before the dict that leaves the renderer. This module installs
the restorer there (`restorer_hook.install`) before the first render.

The wire format is shared by every restore service, so they are interchangeable:

    [4-byte big-endian JSON header length]
    [JSON {"names": [...], "shape": [N,H,W,3], "dtype": "uint8"}]
    [contiguous uint8 (N,H,W,3) RGB]

POST to {url}/restore, response in the same framing, camera order and image size
preserved. A service that lists "bgr" in its /health channel_orders (fixer_server.py) is
sent the renderer's BGR batch as-is with "channel_order": "bgr" in the header and answers in
BGR: the channel flip then happens on its GPU instead of as two strided 25 MB copies here.
Every other service gets RGB, unchanged. Raw bytes rather than base64-in-JSON because the
encoding can cost more than the diffusion itself.

A failed restore RAISES. It is never silently replaced with the raw render: a
run that reports a restorer while showing unrestored pixels is worse than a run
that stops.

FIXER PRESETS. `fixer_h1b16` / `fixer_pretrained` name NVIDIA Fixer (a Cosmos-Predict2
0.6B single-step restorer) with the weights fine-tuned for Odyssey or NVIDIA's released ones. Fixer
needs python 3.12 + cosmos-predict2 and therefore CANNOT be imported into the
renderer process (py3.9), which is the whole reason this out-of-process seam
exists. install_from_env() starts `OdysseyRenderer/fixer/fixer_server.py` in the `fixer` conda
env as a child sidecar, waits for it to report ready ON THE REQUESTED CHECKPOINT,
installs an HttpRestorer pointed at it, and kills it at exit -- including on
SIGKILL, via PDEATHSIG, so a crashed rollout does not strand 10 GB of VRAM.

Knobs, all optional:
  ODYSSEY_FIXER_ROOT     dir holding fixer_server.py, src/, shims/, models/
  ODYSSEY_FIXER_PY       python of the fixer env
  ODYSSEY_FIXER_CKPT     checkpoint override for the chosen preset
  ODYSSEY_FIXER_GPU      CUDA_VISIBLE_DEVICES for the sidecar (required). Fixer runs bf16,
                    so the GPU must support it; a UUID is safer than a bare index,
                    which is read through whatever CUDA_DEVICE_ORDER happens to be set.
  ODYSSEY_FIXER_PORT     0 (default) picks a free port
  ODYSSEY_FIXER_URL      use an already-running sidecar instead of starting one; its
                    /health checkpoint must still match the preset
  ODYSSEY_FIXER_START_TIMEOUT_S   default 900 (a cold load is ~16 s; the margin is for
                    a contended GPU)
Under the benchmark launcher only ODYSSEY_FIXER_START_TIMEOUT_S may be changed: it sets the GPU
itself and refuses ODYSSEY_FIXER_CKPT / _PORT / _URL, which would change the restorer.
"""
import atexit
import json
import logging
import os
import signal
import socket
import subprocess
import time
import urllib.request

import numpy as np

logger = logging.getLogger(__name__)

# This file is <repo>/OdysseyRenderer/odyssey_renderer/omnire/restorer_client.py, so the
# repo root is four levels up.
_REPO_ROOT = os.path.abspath(__file__)
for _ in range(4):
    _REPO_ROOT = os.path.dirname(_REPO_ROOT)
_FIXER_ROOT_DEFAULT = os.path.join(_REPO_ROOT, "OdysseyRenderer", "fixer")
# The Fixer interpreter and GPU have no default: the launcher sets ODYSSEY_FIXER_GPU to the run's
# GPU and the host environment sets ODYSSEY_FIXER_PY (docs/deployment.md, OdysseyRenderer/fixer/README.md).
_FIXER_PY_DEFAULT = None
_FIXER_GPU_DEFAULT = None


def _required(name, value):
    if not value:
        raise RuntimeError(f"{name} is not set: point it at the Fixer environment's python "
                           "(ODYSSEY_FIXER_PY) / the run's GPU (ODYSSEY_FIXER_GPU)")
    return value

# preset -> checkpoint, relative to ODYSSEY_FIXER_ROOT
FIXER_PRESETS = {
    # NVIDIA Fixer fine-tuned by ADRLAB for Odyssey (run h1_b16_e1, iteration 11001): the
    # benchmark's restorer, published in ADRLAB/odyssey-models.
    "fixer_h1b16": "models/finetuned/h1_b16_e1_model_11001.pkl",
    # nvidia/Fixer @ ca20a25b pretrained/pretrained_fixer.pkl, unmodified.
    "fixer_pretrained": "models/pretrained/pretrained_fixer.pkl",
}
RESTORER_CHOICES = tuple(FIXER_PRESETS)


def _env(name, default):
    """os.environ.get, but an EMPTY value means "not set".

    The rollout shells pass these through as `ODYSSEY_FIXER_GPU="${ODYSSEY_FIXER_GPU:-}"`, i.e. as an
    empty string when the caller said nothing. A plain .get() would then hand the sidecar
    CUDA_VISIBLE_DEVICES="" (no GPU at all) or an empty model root, which fails somewhere
    far from the cause.
    """
    return os.environ.get(name, "").strip() or default


def _preset_paths(preset):
    """(fixer root, checkpoint) for a Fixer preset."""
    root = _env("ODYSSEY_FIXER_ROOT", _FIXER_ROOT_DEFAULT)
    ckpt = os.path.realpath(_env("ODYSSEY_FIXER_CKPT", os.path.join(root, FIXER_PRESETS[preset])))
    return root, ckpt


def pack_batch(names, batch, channel_order="rgb"):
    head = {"names": list(names), "shape": list(batch.shape), "dtype": str(batch.dtype)}
    if channel_order != "rgb":
        head["channel_order"] = channel_order
    header = json.dumps(head).encode()
    return len(header).to_bytes(4, "big") + header + np.ascontiguousarray(batch).tobytes()


def unpack_batch(blob):
    n = int.from_bytes(blob[:4], "big")
    header = json.loads(blob[4:4 + n].decode())
    arr = np.frombuffer(blob[4 + n:], dtype=np.dtype(header["dtype"]))
    return header["names"], arr.reshape(header["shape"])


class HttpRestorer:
    """A process restorer (restorer_hook) that forwards the batch to a restore service."""

    def __init__(self, identifier, url, timeout=600.0):
        self.identifier = identifier
        self.url = url.rstrip("/")
        self.timeout = float(timeout)
        self.channel_order = "rgb"

    def health(self):
        with urllib.request.urlopen(self.url + "/health", timeout=30) as r:
            return json.loads(r.read().decode())

    def adopt_health(self, health):
        """Send BGR as-is when the service says it takes it; RGB otherwise."""
        self.channel_order = "bgr" if "bgr" in (health.get("channel_orders") or ()) else "rgb"

    def restore_batch(self, imgs_bgr):
        """BGR uint8 HxWx3 list in, the same list restored, same order and size.

        mtgs.py passes a bare list, so names are positional; the service is
        required to return them in the order it was given.
        """
        if not imgs_bgr:
            raise RuntimeError("cannot send an empty image batch to %s" % self.identifier)
        batch_bgr = np.stack(imgs_bgr)
        if batch_bgr.dtype != np.uint8 or batch_bgr.ndim != 4 or batch_bgr.shape[-1] != 3:
            raise RuntimeError("restorer input must be same-sized uint8 3-channel images, got %r"
                               % (batch_bgr.shape,))
        names = ["cam%d" % i for i in range(batch_bgr.shape[0])]
        bgr = self.channel_order == "bgr"
        # The wire format is RGB unless the service takes BGR (adopt_health).
        wire = batch_bgr if bgr else batch_bgr[..., ::-1]
        req = urllib.request.Request(
            self.url + "/restore", data=pack_batch(names, wire, self.channel_order),
            method="POST", headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            # Server-side compute time, so stage timing can split a restore into model
            # time and transport (the batch is ~50 MB each way at 1080p x3).
            meta = r.headers.get("X-Fixer-Meta")
            meta = json.loads(meta) if meta else {}
            self.last_server_ms = meta.get("ms")
            if bgr and meta.get("channel_order") != "bgr":
                raise RuntimeError("%s did not answer in BGR (X-Fixer-Meta %r); its /health "
                                   "advertised channel_order bgr" % (self.identifier, meta))
            out_names, out = unpack_batch(r.read())
        if list(out_names) != names or tuple(out.shape) != tuple(batch_bgr.shape):
            raise RuntimeError("%s changed camera order or image shape: %r %r"
                               % (self.identifier, out_names, out.shape))
        if bgr:
            # frombuffer is read-only; one contiguous copy keeps the returned images writable.
            return list(out.copy())
        return [np.ascontiguousarray(x[..., ::-1]) for x in out]


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _die_with_parent():
    # Runs in the child between fork and exec: SIGTERM it when the simulator dies,
    # including by SIGKILL, so a crashed rollout does not leave 10 GB on the GPU.
    import ctypes
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG


class _PortTaken(RuntimeError):
    """The sidecar lost the race for its port to another process."""


class FixerSidecar:
    """OdysseyRenderer/fixer/fixer_server.py as a child of this process, owned for its lifetime."""

    def __init__(self, preset):
        root, self.checkpoint = _preset_paths(preset)
        self.preset = preset
        self.root = root
        self.python = _required("ODYSSEY_FIXER_PY", _env("ODYSSEY_FIXER_PY", _FIXER_PY_DEFAULT))
        self.gpu = _required("ODYSSEY_FIXER_GPU", _env("ODYSSEY_FIXER_GPU", _FIXER_GPU_DEFAULT))
        self.proc = None
        self.log_path = None
        for path in (self.checkpoint, self.python, os.path.join(root, "fixer_server.py")):
            if not os.path.exists(path):
                raise RuntimeError("restorer=%s: missing %s" % (preset, path))

    def start(self, timeout):
        """Start the sidecar and wait until IT (not whatever holds the port) is ready.

        _free_port() only finds a port free *now*; the sidecar binds it ~20 s later, after
        loading the model. Rollouts that start together can pick the same port in that
        window: the second sidecar dies on "Address already in use", while /health on that
        URL is answered by the FIRST rollout's sidecar, which this rollout does not own. So a
        ready answer counts only if its pid is our child's, and a lost port race retries on a
        new port.
        """
        fixed = int(_env("ODYSSEY_FIXER_PORT", "0"))
        for attempt in range(1 if fixed else 5):
            try:
                return self._start_once(fixed or _free_port(), timeout)
            except _PortTaken as exc:
                print("[restorer] %s; retrying on another port" % exc, flush=True)
        raise RuntimeError("%s sidecar: port taken on every attempt\n%s"
                           % (self.preset, self.log_tail()))

    def _start_once(self, port, timeout):
        self.url = "http://127.0.0.1:%d" % port
        env = dict(os.environ)
        # The simulator env puts its own py3.9 libs first on LD_LIBRARY_PATH and its tree
        # on PYTHONPATH. Inherited by a py3.12 child those shadow the fixer env's
        # libstdc++ and packages, so the child gets a clean slate instead.
        for key in ("LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME", "TORCH_EXTENSIONS_DIR",
                    "TORCH_CUDA_ARCH_LIST", "CUDA_HOME"):
            env.pop(key, None)
        env.update({
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": self.gpu,
            "PYTHONPATH": os.path.join(self.root, "shims"),
            "FIXER_MODELS_DIR": os.path.join(self.root, "models"),
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
        })
        os.makedirs(os.path.join(self.root, "logs"), exist_ok=True)
        self.log_path = os.path.join(self.root, "logs", "sidecar_%s_%d_%d.log"
                                     % (self.preset, os.getpid(), port))
        cmd = [self.python, os.path.join(self.root, "fixer_server.py"),
               "--host", "127.0.0.1", "--port", str(port),
               "--checkpoint", self.checkpoint, "--src-dir", os.path.join(self.root, "src"),
               "--label", self.preset]
        log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, preexec_fn=_die_with_parent)
        log.close()
        atexit.register(self.stop)
        print("[restorer] starting %s sidecar pid %d on %s gpu=%s ckpt=%s log=%s"
              % (self.preset, self.proc.pid, self.url, self.gpu, self.checkpoint,
                 self.log_path), flush=True)

        client = HttpRestorer(self.preset, self.url)
        deadline = time.time() + timeout
        while True:
            if self.proc.poll() is not None:
                tail = self.log_tail()
                if "Address already in use" in tail or "is in use by another program" in tail:
                    raise _PortTaken("port %d taken by another process" % port)
                raise RuntimeError("%s sidecar exited with %s before becoming ready:\n%s"
                                   % (self.preset, self.proc.returncode, tail))
            try:
                health = client.health()
            except Exception:                                    # noqa: BLE001
                health = None
            if health is not None and health.get("pid") not in (None, self.proc.pid):
                health = None           # another rollout's sidecar on this port -- not ours
            if health is not None:
                if health.get("error"):
                    raise RuntimeError("%s sidecar failed to load: %s\n%s"
                                       % (self.preset, health["error"], self.log_tail()))
                if health.get("ready"):
                    return health
            if time.time() > deadline:
                raise RuntimeError("%s sidecar not ready after %ds:\n%s"
                                   % (self.preset, timeout, self.log_tail()))
            time.sleep(2.0)

    def log_tail(self, n=40):
        try:
            with open(self.log_path, "rb") as f:
                return b"".join(f.readlines()[-n:]).decode(errors="replace")
        except OSError:
            return "(no log)"

    def stop(self):
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _install_fixer(preset):
    from odyssey_renderer.mtgs import restorer_hook

    if restorer_hook.get_restorer() is not None:
        return restorer_hook.get_restorer()
    if os.environ.get('ODYSSEY_RUNTIME_PROFILE'):
        from odyssey_runtime.profile import ModelProfile
        from odyssey_runtime.restorer import SharedRestorer
        profile = ModelProfile.load(os.environ['ODYSSEY_RUNTIME_PROFILE'])
        client = restorer_hook.install(SharedRestorer(preset, profile.data.get('shared_capacity_mb', 64)))
        atexit.register(client.close)
        return client
    timeout = float(_env("ODYSSEY_FIXER_START_TIMEOUT_S", "900"))
    url = os.environ.get("ODYSSEY_FIXER_URL", "").strip()
    _, expected = _preset_paths(preset)
    if url:
        client = HttpRestorer(preset, url)
        health = client.health()
        sidecar = None
    else:
        sidecar = FixerSidecar(preset)
        health = sidecar.start(timeout)
        client = HttpRestorer(preset, sidecar.url)
    # Fine-tuned and pretrained differ ONLY in their weights, so a run labelled with one while
    # restored by another would be silently wrong. Check which checkpoint actually loaded.
    if not health.get("ready") or os.path.realpath(health.get("checkpoint") or "") != expected:
        if sidecar is not None:
            sidecar.stop()
        raise RuntimeError("restorer=%s expects checkpoint %s, sidecar reports %r"
                           % (preset, expected, health))
    client.sidecar = sidecar                  # keep it referenced for the process lifetime
    client.adopt_health(health)
    restorer_hook.install(client)
    print("[restorer] %s installed: url=%s gpu=%s checkpoint=%s timestep=%s proc_hw=%s "
          "wire=%s empty_cache=%s"
          % (preset, client.url, health.get("gpu"), health.get("checkpoint"),
             health.get("timestep"), health.get("proc_hw"), client.channel_order,
             health.get("empty_cache")), flush=True)
    return client


def install_from_env():
    """Install an HTTP restorer as the process restorer, if ODYSSEY_RESTORER asks for one.

    ODYSSEY_RESTORER is one of:
      unset             no restoration (this does nothing)
      "fixer_h1b16"     Fixer fine-tuned for Odyssey, as a managed worker
      "fixer_pretrained" Fixer, NVIDIA's released weights, as a managed worker
      "<id>=<url>"      e.g. harmonizer=http://127.0.0.1:8097
      "<url>"           identifier defaults to the host

    Returns the installed restorer, or None.
    """
    spec = os.environ.get("ODYSSEY_RESTORER", "").strip()
    if not spec:
        return None

    if spec in FIXER_PRESETS:
        return _install_fixer(spec)

    identifier, _, url = spec.partition("=")
    if not url:
        identifier, url = "restorer", identifier
    if not url.startswith("http"):
        raise ValueError("ODYSSEY_RESTORER must be one of %s, or an http url, got %r"
                         % ("|".join(RESTORER_CHOICES), spec))

    from odyssey_renderer.mtgs import restorer_hook

    if restorer_hook.get_restorer() is not None:
        return restorer_hook.get_restorer()

    client = HttpRestorer(identifier, url)
    try:
        h = client.health()
    except Exception as exc:                                     # noqa: BLE001
        raise RuntimeError(
            "ODYSSEY_RESTORER=%s is not reachable (%s: %s). Refusing to start: "
            "falling back to unrestored images while naming a restorer is the "
            "one outcome this must never produce."
            % (spec, type(exc).__name__, exc))
    if not h.get("ready", False):
        raise RuntimeError("restorer at %s is not ready: %r" % (url, h))

    client.adopt_health(h)
    restorer_hook.install(client)
    logger.info("installed out-of-process restorer %s at %s (%s)",
                identifier, url, {k: h[k] for k in list(h)[:6]})
    print("[restorer] %s at %s installed; the render path's restoration hook now calls it"
          % (identifier, url))
    return client
