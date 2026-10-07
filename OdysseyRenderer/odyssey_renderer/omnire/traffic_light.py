"""Portable temporal traffic-light inference, ported from the training code (no training/dataset imports).

Equations: the same as the reconstruction training code's traffic-light node.

Default source-frame validation is upstream-strict. The opt-in active_head_ids
extension resolves only supported heads OUTSIDE the training timestamp window.
Inside that window mask mode renders full natural replay. DB policy belongs to
the caller: this module never infers states or accepts lane/head associations.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from odyssey_renderer.omnire.deform_network import ConditionalDeformNetwork, get_embedder
from odyssey_renderer.mtgs.utils.gaussian_utils import quat_mult, quat_to_rotmat


def _flatten(mapping, prefix=""):
    result = {}
    for key, value in mapping.items():
        path = prefix + key
        if isinstance(value, Mapping):
            result.update(_flatten(value, path + "."))
        else:
            result[path] = value
    return result


def _quat_act(value):
    # The training code's quat_act divides by the norm without an epsilon clamp.
    return value / value.norm(dim=-1, keepdim=True)


def _sh(degree, directions, coefficients):
    # CPU implementation is useful for equation checks and never builds CUDA ops.
    if directions.device.type == "cpu":
        from gsplat.cuda._torch_impl import _spherical_harmonics
        return _spherical_harmonics(degree, directions, coefficients)
    from gsplat.cuda._wrapper import spherical_harmonics
    return spherical_harmonics(degree, directions, coefficients)


class OmniReTrafficLightSubModel(nn.Module):
    """Renderer-compatible, shared-network group of fixed physical heads.

    set_source_frames(mapping, active_head_ids=None) replaces both settings
    atomically. None selects strict all-head semantics. An explicit collection
    must contain exactly mapping.keys(); [] with {} means render no heads
    outside training. Geometry and RGB always share the same original row order.
    """

    MODEL_TYPE = "traffic_light"

    def __init__(self, asset, model_name=None, device=None, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.model_type = self.MODEL_TYPE
        self.config = copy.deepcopy(dict(asset["config"]))
        if self.config.get("schema_version") != 1 or self.config.get("type") not in ("TrafficLightNodes", "OmniReTrafficLightSubModel"):
            raise ValueError("unsupported traffic-light portable schema")
        state = dict(asset["state_dict"])
        metadata = copy.deepcopy(dict(state.pop("_traffic_light_metadata")))
        state = _flatten(state)
        identities = metadata["head_ids"]
        if (metadata.get("version") != 1 or not identities
                or any(not isinstance(h, str) or not h for h in identities)
                or len(set(identities)) != len(identities)
                or set(identities) != set(metadata["heads"])):
            raise ValueError("invalid head metadata")
        self.head_ids = tuple(identities)
        self._head_metadata = metadata["heads"]
        self.source_context = metadata["source_context"]
        self.sh_degree = self.config["sh_degree"]
        self.effective_sh_degree = self.config["effective_sh_degree"]
        self.temporal_sh_mode = self.config["temporal_sh_mode"]
        if (type(self.sh_degree) is not int or not 0 <= self.sh_degree <= 4
                or type(self.effective_sh_degree) is not int
                or not 0 <= self.effective_sh_degree <= self.sh_degree):
            raise ValueError("invalid SH degree")
        if self.temporal_sh_mode not in ("dc", "full") or metadata.get("appearance_mode", "dc") != self.temporal_sh_mode:
            raise ValueError("appearance mode differs from checkpoint")
        self.max_displacement = float(self.config["max_displacement"])
        if not math.isfinite(self.max_displacement) or self.max_displacement < 0:
            raise ValueError("invalid maximum displacement")
        if self.config.get("scale_dim") != 3:
            raise ValueError("only anisotropic 3D Gaussians are supported")
        times = self.config["training_timestamps_us"]
        if (not times or any(type(t) is not int for t in times)
                or any(b <= a for a, b in zip(times, times[1:]))):
            raise ValueError("training timestamps must be increasing integer microseconds")
        self._training_times = tuple(times)
        self._timestamp_frames = {t: index for index, t in enumerate(times)}
        self.num_frames = len(times)
        if (self.source_context != self.config.get("source_context")
                or self.source_context.get("scene_id") != self.config.get("source_scene_id")
                or self.source_context.get("coordinate_frame") != "reconstruction"
                or self.source_context.get("num_frames") != self.num_frames):
            raise ValueError("source context mismatch")
        for identity in self.head_ids:
            entry = self._head_metadata[identity]
            observed = entry["observed_frames"]
            representatives = entry["representatives"]
            if (not isinstance(observed, list) or not observed
                    or any(type(f) is not int or not 0 <= f < self.num_frames for f in observed)
                    or len(set(observed)) != len(observed)):
                raise ValueError(f"{identity}: invalid observations")
            if not isinstance(representatives, Mapping) or any(
                    not isinstance(label, str) or not label or type(f) is not int or f not in observed
                    for label, f in representatives.items()):
                raise ValueError(f"{identity}: invalid representatives")
        self._observed = {h: frozenset(self._head_metadata[h]["observed_frames"]) for h in self.head_ids}
        count = state["gauss_params.means"].shape[0]
        head_count = len(self.head_ids)
        shapes = {"means": (count, 3), "scales": (count, 3), "quats": (count, 4),
                  "opacities": (count, 1), "features_dc": (count, 3),
                  "features_rest": (count, (self.sh_degree + 1)**2 - 1, 3)}
        for name, shape in shapes.items():
            self._register_float("_" + name, state["gauss_params." + name], shape)
        networks = dict(self.config["networks"])
        depth, width, embedding = networks["D"], networks["W"], networks["embed_dim"]
        if type(depth) is not int or depth < 3:
            raise ValueError("deformation depth must be >= 3")
        for key in ("W", "embed_dim", "x_multires", "t_multires"):
            value = networks[key]
            if type(value) is not int or value < (1 if key in ("W", "embed_dim") else 0):
                raise ValueError(f"invalid {key}")
        for key in ("deform_quat", "deform_scale"):
            if type(networks[key]) is not bool:
                raise ValueError(f"invalid {key}")
        self._register_float("instances_size", state["instances_size"], (head_count, 3))
        self._register_float("instances_embedding", state["instances_embedding"], (head_count, embedding))
        self._register_float("installation_trans", state["installation_trans"], (head_count, 3))
        self._register_float("installation_quats", state["installation_quats"], (head_count, 4))
        self._register_float("normalized_timestamps", metadata["normalized_timestamps"], (self.num_frames,))
        if not (self.instances_size > 0).all():
            raise ValueError("head sizes must be positive")
        if not (self.installation_quats.norm(dim=-1) > 0).all() or not (self._quats.norm(dim=-1) > 0).all():
            raise ValueError("quaternions must be nonzero")
        if self.num_frames > 1 and not (self.normalized_timestamps[1:] > self.normalized_timestamps[:-1]).all():
            raise ValueError("normalized timestamps must increase")
        point_ids = state["points_ids"]
        if (point_ids.dtype not in (torch.int32, torch.int64) or tuple(point_ids.shape) != (count, 1)
                or ((point_ids < 0) | (point_ids >= head_count)).any()):
            raise ValueError("invalid point head indices")
        self.register_buffer("point_ids", point_ids.detach().clone().long())
        if any(v.dtype != self._means.dtype for v in self.buffers() if v.is_floating_point()):
            raise ValueError("floating buffer dtypes differ")
        self.deform_network = ConditionalDeformNetwork(**networks).to(dtype=self._means.dtype)
        self._appearance_x, nx = get_embedder(networks["x_multires"], 3)
        self._appearance_t, nt = get_embedder(networks["t_multires"], 1)
        self.appearance_sh_channels = 3 * ((self.sh_degree + 1)**2 if self.temporal_sh_mode == "full" else 1)
        self.appearance_network = nn.Sequential(
            nn.Linear(nx + nt + embedding, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, self.appearance_sh_channels + 1)
        ).to(dtype=self._means.dtype)
        for name in ("deform_network", "appearance_network"):
            weights = {k[len(name) + 1:]: v for k, v in state.items() if k.startswith(name + ".")}
            if any(not torch.isfinite(v).all() or v.dtype != self._means.dtype for v in weights.values()):
                raise ValueError("invalid network tensor values or dtype")
            try:
                getattr(self, name).load_state_dict(weights, strict=True)
            except RuntimeError as error:
                raise ValueError(f"invalid {name} weights: {error}") from error
        self.cur_frame = 0
        self._source_overrides = {}
        self._active_override = None
        self._last_active = self.head_ids
        self._cache = None
        self.eval()
        self.requires_grad_(False)
        if device is not None:
            self.to(device)

    def _register_float(self, name, value, shape):
        if (not torch.is_tensor(value) or tuple(value.shape) != shape
                or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError(f"invalid {name} tensor, expected {shape}")
        self.register_buffer(name, value.detach().clone())

    @property
    def device(self):
        return self._means.device

    @property
    def active_head_ids(self):
        """Heads used by the most recent geometry/RGB/source_frames resolution."""
        return self._last_active

    def set_frame(self, frame):
        if type(frame) is not int:
            raise ValueError("replay frame must be an integer")
        self.cur_frame = frame
        self._cache = None

    def set_source_frames(self, frames, *, active_head_ids=None, allow_unobserved=None):
        """allow_unobserved {head: frame}: that one frame of that head may be unobserved (a label-allowed
        representative, tl_control allow_unobserved, ALLOW_UNOBSERVED_SCENES only). It must still be a training frame."""
        if self.training:
            raise ValueError("source overrides are disabled during training")
        allowed = dict(allow_unobserved or {})
        for identity in allowed:
            if identity not in self._observed:
                raise KeyError(f"unknown head {identity!r}")
        validated = {}
        for identity, frame in frames.items():
            if identity not in self._observed:
                raise KeyError(f"unknown head {identity!r}")
            if type(frame) is not int or (
                    frame not in self._observed[identity]
                    and not (allowed.get(identity) == frame and 0 <= frame < self.num_frames)):
                raise ValueError(f"{identity}: source frame must be observed")
            validated[identity] = frame
        selected = None
        if active_head_ids is not None:
            if isinstance(active_head_ids, (str, bytes)):
                raise ValueError("active_head_ids must be a collection of head IDs")
            requested = list(active_head_ids)
            for identity in requested:
                if identity not in self._observed:
                    raise KeyError(f"unknown head {identity!r}")
            if len(set(requested)) != len(requested) or set(requested) != set(validated):
                raise ValueError("mask and source mapping must name exactly the same unique heads")
            selected = tuple(h for h in self.head_ids if h in validated)
        # Commit only after the whole request validated; always replace both.
        self._source_overrides = validated
        self._active_override = selected
        self._cache = None

    def clear_source_frames(self):
        self._source_overrides = {}
        self._active_override = None
        self._cache = None

    def set_representative_states(self, states):
        frames = {}
        for identity, label in states.items():
            if identity not in self._head_metadata:
                raise KeyError(f"unknown head {identity!r}")
            representatives = self._head_metadata[identity]["representatives"]
            if label not in representatives:
                raise ValueError(f"{identity}: no representative for {label!r}")
            frames[identity] = representatives[label]
        self.set_source_frames(frames)

    def train(self, mode=True):
        if mode:
            self.clear_source_frames()
        return super().train(mode)

    def _apply(self, fn):
        self._cache = None
        return super()._apply(fn)

    def load_state_dict(self, state_dict, strict=True):
        self._cache = None
        return super().load_state_dict(state_dict, strict=strict)

    def _resolve(self, timestamp):
        frame = self.cur_frame
        if timestamp is not None:
            if isinstance(timestamp, bool) or not isinstance(timestamp, Integral):
                raise ValueError("timestamp must be integer microseconds")
            timestamp = int(timestamp)
            if timestamp in self._timestamp_frames:
                frame = self._timestamp_frames[timestamp]
            elif timestamp < self._training_times[0]:
                frame = -1
            elif timestamp > self._training_times[-1]:
                frame = self.num_frames
            else:
                raise ValueError("natural replay requires an exact saved training timestamp")
        in_window = 0 <= frame < self.num_frames
        active = self.head_ids
        overrides = {} if self.training else self._source_overrides
        if self._active_override is not None and not self.training:
            if in_window:
                # Explicit DB mode may never replace in-training natural replay.
                overrides = {}
            else:
                active = self._active_override
        frames = tuple(overrides.get(h, frame) for h in active)
        if any(f < 0 or f >= self.num_frames for f in frames):
            raise ValueError("outside the learned window every active head requires an observed source frame")
        self._last_active = active
        return active, frames

    def source_frames(self, timestamp=None):
        """Source indices in active_head_ids order; default is all upstream heads."""
        _, frames = self._resolve(timestamp)
        return torch.tensor(frames, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _evaluate(self, timestamp):
        active, frames = self._resolve(timestamp)
        # Version counters also prevent stale results after standard state reloads
        # or explicit weight edits by diagnostics; no CUDA readback is needed.
        tensors = (*self.buffers(), *self.parameters())
        # Inference tensors have no version counter. Recompute rather than
        # crashing or retaining stale values after an untracked in-place edit.
        versioned = not any(torch.is_inference(t) for t in tensors)
        versions = tuple(t._version for t in tensors) if versioned else None
        key = (active, frames, self.device, self._means.dtype, versions) if versioned else None
        if key is not None and self._cache is not None and self._cache[0] == key:
            return self._cache[1], self._cache[2]
        indices = self.point_ids[:, 0]
        if active == self.head_ids:
            rows = slice(None)
            selected_ids = indices
        else:
            head_mask = torch.tensor([h in active for h in self.head_ids], device=self.device, dtype=torch.bool)
            rows = head_mask[indices]
            selected_ids = indices[rows]
        means = self._means[rows]
        if means.shape[0] == 0:
            geometry = {"means": means, "scales": self._scales[rows],
                        "quats": self._quats[rows], "opacities": self._opacities[rows].squeeze(-1)}
            coefficients = self._features_rest.new_empty((0, (self.sh_degree + 1)**2, 3))
        else:
            source_by_head = torch.zeros(len(self.head_ids), dtype=torch.long, device=self.device)
            head_indices = [self.head_ids.index(h) for h in active]
            source_by_head[head_indices] = torch.tensor(frames, dtype=torch.long, device=self.device)
            x = means.detach() / self.instances_size[selected_ids].clamp_min(1e-6) * 2
            times = self.normalized_timestamps[source_by_head[selected_ids]].reshape(-1, 1)
            condition = self.instances_embedding[selected_ids]
            residual = self.appearance_network(torch.cat(
                (self._appearance_x(x), self._appearance_t(times), condition), -1))
            xyz, quat, scale = self.deform_network(x, times, condition)
            local_means = means + self.max_displacement * torch.tanh(xyz)
            local_quats = _quat_act(self._quats[rows])
            if quat is not None:
                local_quats = _quat_act(local_quats + .05 * torch.tanh(quat))
            installation = self.installation_quats[selected_ids]
            # The training code's _vendored_ops normalizes installation quats before rotation.
            rotation = quat_to_rotmat(F.normalize(installation, dim=-1))
            world_means = torch.bmm(rotation, local_means.unsqueeze(-1)).squeeze(-1)
            world_means = world_means + self.installation_trans[selected_ids]
            world_quats = _quat_act(quat_mult(installation, local_quats))
            scales = self._scales[rows].exp()
            if scale is not None:
                scales = scales * (.2 * torch.tanh(scale)).exp()
            coefficients = torch.cat((self._features_dc[rows, None, :], self._features_rest[rows]), 1)
            if self.temporal_sh_mode == "full":
                coefficients = coefficients + residual[:, :-1].reshape_as(coefficients)
            else:
                coefficients = torch.cat(((coefficients[:, 0] + residual[:, :3])[:, None],
                                          coefficients[:, 1:]), 1)
            geometry = {"means": world_means, "quats": world_quats, "scales": scales,
                        "opacities": (self._opacities[rows] + residual[:, -1:]).sigmoid().squeeze(-1)}
        self._cache = (key, geometry, coefficients) if key is not None else None
        return geometry, coefficients

    def get_global_gaussians(self, quat=None, trans=None, timestamp=None, **kwargs):
        if quat is not None or trans is not None:
            raise ValueError("traffic-light installations cannot receive actor pose commands")
        return self._evaluate(timestamp)[0]

    @torch.no_grad()
    def get_gaussian_rgbs(self, camera_to_worlds, timestamp=None, device=None, **kwargs):
        if device is not None:
            requested = torch.device(device)
            if requested.type == "cuda" and requested.index is None:
                requested = torch.device("cuda", torch.cuda.current_device())
            if requested != self.device:
                raise ValueError("move the model to the requested device before rendering")
        geometry, coefficients = self._evaluate(timestamp)
        if coefficients.shape[0] == 0:
            return self._means.new_empty((0, 3))
        camera = torch.as_tensor(camera_to_worlds, device=self.device, dtype=self._means.dtype)
        if tuple(camera.shape) == (1, 4, 4):
            camera = camera[0]
        if tuple(camera.shape) != (4, 4) or not torch.isfinite(camera).all():
            raise ValueError("camera must be one finite 4x4 camera-to-world matrix")
        if self.sh_degree == 0:
            return coefficients[:, 0].sigmoid()
        directions = geometry["means"].detach() - camera.detach()[:3, 3]
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return (_sh(self.effective_sh_degree, directions, coefficients) + .5).clamp(0, 1)
