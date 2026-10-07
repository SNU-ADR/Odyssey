"""Import-only stub for flash_attn.

imaginaire.networks.qwen2_5_vl asserts flash_attn is importable at module load.
That module backs the Qwen guardrail / text-encoder path, which Fixer disables
(`config.guardrail_config.enabled = False`) and the `nocond` model never calls.
Every symbol here raises if actually invoked, so a real dependency on flash
attention fails loudly instead of silently returning something wrong.
"""
__version__ = "2.7.4.post1"


def _unavailable(name):
    def _f(*args, **kwargs):
        raise NotImplementedError(
            f"flash_attn.{name} was called, but only an import stub is installed. "
            "Install the real flash-attn if this code path is genuinely needed."
        )
    return _f


flash_attn_varlen_func = _unavailable("flash_attn_varlen_func")
flash_attn_func = _unavailable("flash_attn_func")
