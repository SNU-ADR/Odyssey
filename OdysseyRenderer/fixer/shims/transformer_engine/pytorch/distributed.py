"""Minimal stand-ins. imaginaire.utils.graph imports these for CUDA-graph capture,
which the single-GPU inference path never enters."""
def get_all_rng_states():
    return {}

def graph_safe_rng_available():
    return False


from contextlib import contextmanager


@contextmanager
def activation_recompute_forward(activation_recompute=False, recompute_phase=False, **kwargs):
    """No-op. TE uses this to mark the recompute phase; inference never recomputes."""
    yield


def in_fp8_activation_recompute_phase():
    return False


def prepare_te_modules_for_fsdp(*args, **kwargs):
    return None
