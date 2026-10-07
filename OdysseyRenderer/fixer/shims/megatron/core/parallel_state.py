"""Single-process defaults for the six functions cosmos_predict2 uses."""

def is_initialized():
    return False

def get_data_parallel_world_size():
    return 1

def get_data_parallel_rank():
    return 0

def get_context_parallel_group():
    return None

def get_context_parallel_world_size():
    return 1

def get_context_parallel_rank():
    return 0

def get_tensor_model_parallel_world_size():
    return 1

def get_tensor_model_parallel_rank():
    return 0
