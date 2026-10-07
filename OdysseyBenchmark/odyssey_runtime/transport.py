"""Private inherited socket and anonymous mmap; one outstanding request owns the buffer.

Control messages use pickle only between the parent and its own child (no network listener).
The NumPy metadata types/dtypes survive exactly; images never enter the control message.
"""

import atexit
import mmap
import os
from pathlib import Path
import pickle
import signal
import socket
import struct
import subprocess
import tempfile
import threading
import time

MAX_CONTROL = 16 * 1024 * 1024


def send(sock, message):
    body = pickle.dumps(message, protocol=4)
    if len(body) > MAX_CONTROL:
        raise ValueError("control message exceeds limit")
    sock.sendall(struct.pack("!I", len(body)) + body)


def receive(sock, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout

    def exact(n):
        out = bytearray(n)
        view = memoryview(out)
        while view:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("worker response deadline expired")
                sock.settimeout(remaining)
            size = sock.recv_into(view)
            if not size:
                raise EOFError("worker control socket closed")
            view = view[size:]
        return out

    size = struct.unpack("!I", exact(4))[0]
    if size > MAX_CONTROL:
        raise ValueError("invalid control message size")
    return pickle.loads(exact(size))


class SharedWorker:
    def __init__(
        self, python, factory, config, capacity, log_path, env=None, timeout=300
    ):
        if capacity <= 0 or timeout <= 0:
            raise ValueError("positive capacity and timeout required")
        self.capacity = int(capacity)
        self.timeout = float(timeout)
        self.seq = 0
        self._closed = False
        self.lock = threading.Lock()
        self.proc = None
        self.file = tempfile.TemporaryFile(prefix="odyssey-runtime-", dir="/dev/shm")
        self.file.truncate(self.capacity)
        self.buffer = mmap.mmap(self.file.fileno(), self.capacity)
        self.sock, child = socket.socketpair()
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        child_env = dict(os.environ if env is None else env)
        root = str(Path(__file__).resolve().parents[1])       # OdysseyBenchmark/: odyssey_runtime, odyssey_bridge
        child_env["PYTHONPATH"] = root + os.pathsep + child_env.get("PYTHONPATH", "")
        command = [
            str(python),
            "-m",
            "odyssey_runtime.worker",
            str(child.fileno()),
            str(self.file.fileno()),
            str(self.capacity),
            factory,
            str(os.getpid()),
        ]
        try:
            with log_path.open("wb") as log:
                self.proc = subprocess.Popen(
                    command,
                    pass_fds=(child.fileno(), self.file.fileno()),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                    start_new_session=True,
                )
            child.close()
            send(self.sock, config)
            ready = receive(self.sock, self.timeout)
            if not ready.get("ready"):
                raise RuntimeError(f"worker startup failed: {ready}; log={log_path}")
            self.ready = ready.get("metadata", {})
        except BaseException:
            child.close()
            self.close()
            raise
        atexit.register(self.close)

    def request(self, message, payload=None):
        with self.lock:
            if self._closed:
                raise RuntimeError("worker is closed or failed")
            n = 0 if payload is None else len(payload)
            if n > self.capacity:
                raise ValueError("payload exceeds shared-memory capacity")
            if n:
                self.buffer[:n] = payload
            self.seq += 1
            try:
                send(self.sock, dict(message, seq=self.seq, payload_size=n))
                reply = receive(self.sock, self.timeout)
                if reply.get("seq") != self.seq:
                    raise RuntimeError("worker response sequence mismatch")
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply["result"]
            except BaseException as exc:
                self.close()
                if isinstance(exc, socket.timeout):
                    raise TimeoutError("worker response deadline expired") from exc
                if isinstance(
                    exc, (EOFError, ConnectionError, OSError)
                ) and not isinstance(exc, TimeoutError):
                    raise RuntimeError(
                        "worker exited or control connection failed"
                    ) from exc
                raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.sock.close()
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGTERM)
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait()
        self.buffer.close()
        self.file.close()
        atexit.unregister(self.close)
