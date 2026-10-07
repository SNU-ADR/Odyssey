import numpy as np
import pytest
from odyssey_runtime.restorer import validate_batch


def test_restorer_batch_preserves_camera_order_and_bytes():
    a = [np.full((8, 9, 3), v, np.uint8) for v in (11, 22, 33)]
    out = validate_batch(a)
    assert out.shape == (3, 8, 9, 3)
    assert [x[0, 0, 0] for x in out] == [11, 22, 33]


@pytest.mark.parametrize(
    "batch",
    [
        [],
        [np.zeros((2, 2, 3), np.float32)],
        [np.zeros((2, 2, 3), np.uint8), np.zeros((3, 2, 3), np.uint8)],
    ],
)
def test_invalid_batch_fails_before_rpc(batch):
    with pytest.raises(ValueError):
        validate_batch(batch)
