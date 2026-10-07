"""Import-only stub for megatron. cosmos_predict2 imports
`from megatron.core import parallel_state` at module level but only calls six
single-process-trivial functions from it. Installing real megatron.core instead
drags in transformer_engine's compiled internals, which is the thing we are
avoiding."""
