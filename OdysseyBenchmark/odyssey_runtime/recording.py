"""Bounded optional recording; background failures are re-raised on the owner thread."""

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path


class Recorder:
    def __init__(self, max_pending=4):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.limit = max_pending
        self.pending = deque()
        self.error = None
        self.closed = False
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="odyssey-recorder")

    @staticmethod
    def _write(path, data):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def _wait_one(self):
        try:
            self.pending.popleft().result()
        except BaseException as exc:
            self.error = exc
            raise

    def submit_bytes(self, path, data):
        if self.error:
            raise self.error
        if self.closed:
            raise RuntimeError("recorder is closed")
        if len(self.pending) >= self.limit:
            self._wait_one()
        self.pending.append(self.pool.submit(self._write, path, bytes(data)))

    def flush(self):
        if self.error:
            raise self.error
        while self.pending:
            self._wait_one()

    def close(self):
        try:
            self.flush()
        finally:
            self.closed = True
            self.pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
