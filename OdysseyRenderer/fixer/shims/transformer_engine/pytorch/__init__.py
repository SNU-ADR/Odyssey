import torch
from torch import nn

from . import attention  # noqa: F401
from . import distributed  # noqa: F401
from . import module  # noqa: F401


class RMSNorm(nn.Module):
    """Matches transformer_engine.pytorch.RMSNorm with zero_centered_gamma=False.

    TE normalizes in fp32 and casts back to the input dtype; mirrored here so a
    bf16 forward pass gives the same result as the compiled kernel.
    """

    def __init__(self, hidden_size, eps=1e-5, sequence_parallel=False,
                 params_dtype=None, zero_centered_gamma=False, device=None, **kwargs):
        super().__init__()
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.weight = nn.Parameter(
            torch.empty(hidden_size, dtype=params_dtype or torch.get_default_dtype(), device=device)
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.fill_(0.0 if self.zero_centered_gamma else 1.0)

    def forward(self, x):
        in_dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        w = self.weight.float()
        if self.zero_centered_gamma:
            w = w + 1.0
        return (xf * w).to(in_dtype)


LayerNorm = nn.LayerNorm
