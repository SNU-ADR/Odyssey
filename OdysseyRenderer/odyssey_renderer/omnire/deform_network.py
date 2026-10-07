"""OmniRe's ConditionalDeformNetwork, vendored verbatim.

Transcribed from drivestudio `models/modules.py` (get_embedder / Embedder at
:305-366, ConditionalDeformNetwork at :411-457). It is copied rather than
re-derived so that the numbers are upstream's; the equivalence test loads real
checkpoint weights into BOTH this class and upstream's and compares outputs.

Two transcription traps, both preserved deliberately:

  * `self.skips = [D // 2]`, but the ModuleList is `[Linear(input_ch, W)]` plus a
    comprehension over `range(D - 1)`, so list index k corresponds to
    comprehension index i = k - 1. The WIDE layer therefore lands at
    `linear[D//2 + 1]` while `forward` concatenates after `linear[D//2]`. It
    lines up only by that off-by-one. Do not "fix" either half alone.
  * `forward` returns `(d_xyz, rotation, scaling)` while the caller unpacks
    `delta_xyz, delta_quat, delta_scale`. Consistent, easy to mis-transcribe.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=1):
    if i == -1:
        return nn.Identity(), 3
    embed_kwargs = {
        'include_input': True,
        'input_dims': i,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }
    eo = Embedder(**embed_kwargs)
    return (lambda x, eo=eo: eo.embed(x)), eo.out_dim


class ConditionalDeformNetwork(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, embed_dim=10,
                 x_multires=10, t_multires=10,
                 deform_quat=True, deform_scale=True):
        super().__init__()
        self.D = D
        self.W = W
        self.deform_quat = deform_quat
        self.deform_scale = deform_scale
        self.skips = [D // 2]

        self.embed_time_fn, time_input_ch = get_embedder(t_multires, 1)
        self.embed_fn, xyz_input_ch = get_embedder(x_multires, 3)
        self.input_ch = xyz_input_ch + time_input_ch + embed_dim

        self.linear = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [
                nn.Linear(W, W) if i not in self.skips
                else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)]
        )
        self.gaussian_warp = nn.Linear(W, 3)
        if self.deform_quat:
            self.gaussian_rotation = nn.Linear(W, 4)
        if self.deform_scale:
            self.gaussian_scaling = nn.Linear(W, 3)

    def forward(self, x, t, condition):
        t_emb = self.embed_time_fn(t)
        x_emb = self.embed_fn(x)
        h = torch.cat([x_emb, t_emb, condition], dim=-1)
        for i, l in enumerate(self.linear):
            h = self.linear[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([x_emb, t_emb, condition, h], -1)
        d_xyz = self.gaussian_warp(h)
        scaling, rotation = None, None
        if self.deform_scale:
            scaling = self.gaussian_scaling(h)
        if self.deform_quat:
            rotation = self.gaussian_rotation(h)
        return d_xyz, rotation, scaling
