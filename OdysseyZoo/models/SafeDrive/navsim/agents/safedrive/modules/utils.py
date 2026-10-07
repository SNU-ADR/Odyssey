from __future__ import annotations

# file: fp16_utils.py
import functools
import inspect
from typing import Any, Mapping, Tuple, Union

import torch
from torch import amp

TensorLike = Union[torch.Tensor, Any]

def _cast_tree(x: Any, to_dtype: torch.dtype) -> Any:
    """Cast only the float tensors inside a dict / list / tuple."""
    if isinstance(x, torch.Tensor):
        if torch.is_floating_point(x) and x.dtype != to_dtype:
            return x.to(to_dtype)
        return x
    elif isinstance(x, Mapping):
        return x.__class__({k: _cast_tree(v, to_dtype) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        casted = [_cast_tree(v, to_dtype) for v in x]
        return x.__class__(casted)
    return x

def _select_argnames(fn, apply_to: Tuple[str, ...] | None) -> Tuple[str, ...]:
    if apply_to is not None:
        return tuple(apply_to)
    # default: every positional argument except self
    sig = inspect.signature(fn)
    names = [p.name for p in sig.parameters.values()
             if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    return tuple(n for n in names if n != "self")

def auto_fp16(
    apply_to: Tuple[str, ...] | None = None,
    out_fp32: bool = False,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,  # bfloat16 is usually the better choice on newer GPUs
):
    """
    Replacement for mmcv.runner.auto_fp16.
    - apply_to: argument names to cast; None means every tensor argument except self.
    - out_fp32: cast the return value back to float32.
    - use_amp: run under torch.amp.autocast (recommended).
    - amp_dtype: autocast dtype (torch.float16 | torch.bfloat16).
    Usage:
        class M(nn.Module):
            fp16_enabled = True
            @auto_fp16(apply_to=('x', 'y'), out_fp32=False, use_amp=True, amp_dtype=torch.bfloat16)
            def forward(self, x, y, mask=None):
                ...
    """
    def decorator(fn):
        argnames = _select_argnames(fn, apply_to)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # honour self.fp16_enabled, defaulting to True when absent
            fp16_on = True
            if len(args) > 0 and hasattr(args[0], "fp16_enabled"):
                fp16_on = bool(getattr(args[0], "fp16_enabled"))

            if not torch.cuda.is_available():
                fp16_on = False  # disabled on CPU

            # bind by signature so the mapping is safe
            bound = inspect.signature(fn).bind_partial(*args, **kwargs)
            bound.apply_defaults()

            if fp16_on:
                for name in argnames:
                    if name in bound.arguments:
                        bound.arguments[name] = _cast_tree(bound.arguments[name], amp_dtype)

            call_args = bound.args
            call_kwargs = bound.kwargs

            if fp16_on and use_amp:
                with amp.autocast(device_type="cuda", dtype=amp_dtype):
                    out = fn(*call_args, **call_kwargs)
            else:
                out = fn(*call_args, **call_kwargs)

            if out_fp32 and fp16_on:
                out = _cast_tree(out, torch.float32)
            return out
        return wrapper
    return decorator


def force_fp32(
    apply_to: Tuple[str, ...] | None = None,
    out_fp16: bool = False,
    amp_enabled_check: bool = True,
):
    """
    Replacement for mmcv.runner.force_fp32.
    - Forces the named arguments to float32 and runs with autocast disabled.
    - out_fp16 casts the return value back to float16.
    """
    def decorator(fn):
        argnames = _select_argnames(fn, apply_to)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = inspect.signature(fn).bind_partial(*args, **kwargs)
            bound.apply_defaults()

            for name in argnames:
                if name in bound.arguments:
                    bound.arguments[name] = _cast_tree(bound.arguments[name], torch.float32)

            call_args = bound.args
            call_kwargs = bound.kwargs

            with amp.autocast(device_type="cuda", enabled=False):
                out = fn(*call_args, **call_kwargs)

            if out_fp16:
                out = _cast_tree(out, torch.float16)
            return out
        return wrapper
    return decorator
