import pytest
from odyssey_runtime.recording import Recorder


def test_flush_persists_exact_bytes(tmp_path):
    with Recorder(max_pending=2) as r:
        for i in range(9):
            r.submit_bytes(tmp_path / f"{i}.bin", bytes([i]) * 100)
        r.flush()
        assert all(
            (tmp_path / f"{i}.bin").read_bytes() == bytes([i]) * 100 for i in range(9)
        )


def test_background_write_failure_reaches_owner(tmp_path):
    p = tmp_path / "not_directory"
    p.write_bytes(b"file")
    r = Recorder(max_pending=1)
    r.submit_bytes(p / "child", b"bad")
    with pytest.raises(OSError):
        r.flush()
    with pytest.raises(OSError):
        r.close()
