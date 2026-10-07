"""One actor identity containing all its valid local deformation networks."""
import torch
from odyssey_renderer.mtgs.gaussian_model.rigid_object import RigidPortableSubModel
from odyssey_renderer.omnire.deformable_object import OmniReDeformableSubModel


class OmniReBlendedDeformableSubModel(RigidPortableSubModel):
    MODEL_TYPE = 'deformable'

    def __init__(self, asset, **kwargs):
        super().__init__(asset=asset, **kwargs)
        self.parts = torch.nn.ModuleList([
            OmniReDeformableSubModel(asset=p, model_name=self.model_name+'_'+str(i))
            for i, p in enumerate(self.config.blend_parts)])
        self._active_parts = []

    def get_global_gaussians(self, quat=None, trans=None, timestamp=None, **kwargs):
        self._active_parts = []
        ts = self.config.log_timestamps
        t = float(timestamp)
        if t < float(ts[0]) or t > float(ts[-1]):
            return None
        hi = min(int(torch.searchsorted(ts, ts.new_tensor(t))), len(ts)-1)
        lo = max(hi-1, 0)
        u = 0 if hi == lo else (t-float(ts[lo]))/float(ts[hi]-ts[lo])
        table = self.config.blend_weights
        weights = table[lo]*(1-u) + table[hi]*u
        quat, trans, _ = self._decide_global_pose(quat, trans, timestamp)
        if quat is None or trans is None:
            return None
        collected = []
        for i, part in enumerate(self.parts):
            w = float(weights[i])
            if w <= 0:
                continue
            # Even driven actors never extrapolate a local network's domain.
            pts = part.config.log_timestamps
            if t < float(pts[0]) or t > float(pts[-1]):
                continue
            gs = part.get_global_gaussians(quat=quat, trans=trans, timestamp=timestamp)
            if gs is not None:
                collected.append((gs, w))
                self._active_parts.append(part)
        if not collected:
            return None
        total = sum(w for _, w in collected)
        for gs, w in collected:
            alpha = gs['opacities'].clamp(max=1-1e-7)
            gs['opacities'] = -torch.expm1(torch.log1p(-alpha)*(w/total))
        return {k: torch.cat([gs[k] for gs, _ in collected]) for k in collected[0][0]}

    def get_gaussian_rgbs(self, camera_to_worlds, timestamp, device=None):
        return torch.cat([p.get_gaussian_rgbs(camera_to_worlds, timestamp, device)
                          for p in self._active_parts])
