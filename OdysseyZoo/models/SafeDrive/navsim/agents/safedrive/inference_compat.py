"""Numeric helpers: a masked scatter without host syncs (SafeDrive_Model) and a context manager
that turns TF32 off (BEV encoder point sampling)."""
from contextlib import contextmanager
import torch


def scatter_valid(destination, dim, index, source, valid):
    """Scatter unique valid query IDs; padding writes go to a discarded slot.

    SafeDrive's top-k selection supplies unique valid query IDs in range.
    A separate padding slot avoids CPU synchronization/boolean compaction.
    Invalid values (including NaN) cannot overwrite real query zero.
    """
    padding_shape = list(destination.shape)
    padding_shape[dim] = 1
    padded = torch.cat((destination, destination.new_zeros(padding_shape)), dim=dim)
    safe_index = index.masked_fill(~valid, destination.shape[dim])
    padded.scatter_(dim, safe_index, source)
    return padded.narrow(dim, 0, destination.shape[dim]).contiguous()


@contextmanager
def tf32_disabled():
    """Temporarily disable both TF32 backends and restore each independently."""
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
