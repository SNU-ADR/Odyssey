import os
from pathlib import Path
import sys
import time
import pytest
from odyssey_runtime.transport import SharedWorker


@pytest.fixture
def worker(tmp_path):
    w = SharedWorker(
        sys.executable,
        "tests.runtime.worker_probe:Probe",
        {},
        128,
        tmp_path / "worker.log",
        timeout=1,
    )
    yield w
    w.close()


def test_real_child_shares_buffer_and_sequences_requests(worker):
    assert worker.request({"op": "reverse"}, b"abcd") == {"size": 4}
    assert bytes(worker.buffer[:4]) == b"dcba"
    assert worker.request({"op": "reverse"}, b"123") == {"size": 3}
    assert bytes(worker.buffer[:3]) == b"321"


def test_oversized_payload_rejected_without_poisoning_worker(worker):
    with pytest.raises(ValueError, match="capacity"):
        worker.request({"op": "reverse"}, b"x" * 129)
    assert worker.request({"op": "reverse"}, b"x") == {"size": 1}


@pytest.mark.parametrize(
    "op,exc", [("error", RuntimeError), ("exit", RuntimeError), ("sleep", TimeoutError)]
)
def test_worker_failures_fail_closed(worker, op, exc):
    with pytest.raises(exc):
        worker.request({"op": op})
    with pytest.raises(RuntimeError):
        worker.request({"op": "reverse"}, b"x")
    assert worker.proc.poll() is not None


def test_close_is_idempotent_and_child_is_reaped(worker):
    worker.close()
    worker.close()
    assert worker.proc.poll() is not None
