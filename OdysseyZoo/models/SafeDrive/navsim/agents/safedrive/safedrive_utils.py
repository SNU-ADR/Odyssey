"""Pure helpers used by the SafeDrive model.

Split out of safedrive_model.py: functions and small nn.Module blocks that carry
no model state, so the model file stays focused on SafeDrive_Model itself.
"""
import copy
from typing import Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters


class FrozenModule(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        with torch.no_grad():
            return self.module(x) * 0.0

def safe_log(x, eps=1e-6):
    """Safe log: guards against log(0) and NaN/Inf while keeping the negative penalty."""
    return torch.log(x.clamp_min(eps))

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

def calc_projection_mats(
    matrices,
    crop_offset: tuple = (28, 0),
    resize_shape: tuple = (512, 256),
):
    """Recompute lidar2img and post_rots / post_trans on the fly.

    Cached post_rots / post_trans may carry an older single resize factor, so they are
    rebuilt from the configured crop_offset and resize_shape.
    Everything runs in float32 under autocast(enabled=False).

    Args:
        matrices: features['matrices'].
            Non-temporal: List[6] of [B, N_cam, ...] tensors.
            Temporal:     List[T] of (List[6] of [B, N_cam, ...] tensors).
        crop_offset:  (h_crop, w_crop) — default (28, 0)
        resize_shape: (W, H) of final image — default (512, 256)
    Returns:
        lidar2img  : [B, T, N_cam, 4, 4] float32
        post_rots  : [B, N_cam, 3, 3] float32, shared by every frame
        post_trans : [B, N_cam, 3]        float32
    """
    if not isinstance(matrices[0], (list, tuple)):
        matrices = [matrices]   # non-temporal → T=1

    # post matrix recomputed from the config, shared by every frame and camera
    cropped_h = 1080 - 2 * crop_offset[0]
    cropped_w = 1920 - 2 * crop_offset[1]
    resize_x  = resize_shape[0] / cropped_w   # 512/1920 ≈ 0.2667
    resize_y  = resize_shape[1] / cropped_h   # 256/1024 = 0.25

    pr_np = np.eye(3, dtype=np.float32)
    pr_np[0, 0] = resize_x
    pr_np[1, 1] = resize_y

    pt_np = np.zeros(3, dtype=np.float32)
    pt_np[0] = -crop_offset[1] * resize_x   # 0
    pt_np[1] = -crop_offset[0] * resize_y   # -7.0

    # 4x4 post for lidar2img: it multiplies [u*Z, v*Z, Z, 1], so the translation
    # belongs in the Z column to stay depth-independent after the division.
    post4_np = np.eye(4, dtype=np.float32)
    post4_np[0, 0] = resize_x
    post4_np[1, 1] = resize_y
    post4_np[0, 2] = pt_np[0]   # 0         (Z-column)
    post4_np[1, 2] = pt_np[1]   # -7.0      (Z-column)

    frame_l2i = []
    for frame_mats in matrices:
        rots, trans, intrins = frame_mats[0], frame_mats[1], frame_mats[2]
        # rots:   [B, N_cam, 3, 3]  sensor2lidar rotation (cam→lidar)
        # trans:  [B, N_cam, 3]
        # intrins:[B, N_cam, 3, 3]
        B, N = rots.shape[:2]
        dev  = rots.device
        BN   = B * N

        with torch.cuda.amp.autocast(enabled=False):
            rots_f    = rots.float()
            trans_f   = trans.float()
            intrins_f = intrins.float()

            rot_inv  = rots_f.transpose(-1, -2)                            # (B, N, 3, 3)
            tran_inv = -(rot_inv @ trans_f.unsqueeze(-1)).squeeze(-1)      # (B, N, 3)

            lidar2cam = torch.eye(4, dtype=torch.float32, device=dev).view(1,1,4,4).expand(B,N,4,4).clone()
            lidar2cam[:, :, :3, :3] = rot_inv
            lidar2cam[:, :, :3,  3] = tran_inv

            cam2img = torch.eye(4, dtype=torch.float32, device=dev).view(1,1,4,4).expand(B,N,4,4).clone()
            cam2img[:, :, :3, :3] = intrins_f

            post4 = torch.tensor(post4_np, dtype=torch.float32, device=dev).view(1,1,4,4).expand(B,N,4,4)

            l2i = (post4.reshape(BN,4,4)
                   @ cam2img.view(BN,4,4)
                   @ lidar2cam.view(BN,4,4)).view(B, N, 4, 4)   # float32
        frame_l2i.append(l2i)

    lidar2img = torch.stack(frame_l2i, dim=1)   # (B, T, N_cam, 4, 4)

    B, N = matrices[0][0].shape[:2]
    dev  = matrices[0][0].device
    post_rots  = torch.tensor(pr_np, dtype=torch.float32, device=dev).view(1,1,3,3).expand(B,N,-1,-1).contiguous()
    post_trans = torch.tensor(pt_np, dtype=torch.float32, device=dev).view(1,1,3).expand(B,N,-1).contiguous()

    return lidar2img, post_rots, post_trans

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class GlobalResponseNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
        self.eps   = eps
    def forward(self, x):           # x: (B,C,H,W)
        gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)       # (B,C,1,1)
        nx = x * (gx / (gx.mean(dim=1, keepdim=True) + self.eps))
        return self.gamma[:,None,None] * nx + self.beta[:,None,None]

class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv  = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm    = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4*dim, 1)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Conv2d(4*dim, dim, 1)
        self.grn     = GlobalResponseNorm(dim)
        self.drop    = timm.layers.DropPath(drop_path) if drop_path > 0 else nn.Identity()
    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.pwconv1(x); x = self.act(x); x = self.pwconv2(x)
        x = self.grn(x)
        x = self.drop(x)
        return x + shortcut

class LayerNorm2d(nn.LayerNorm):
    # ConvNeXt style LayerNorm that works on (N, C, H, W)
    def __init__(self, dim, eps=1e-6):
        super().__init__(dim, eps=eps)

    def forward(self, x):
        # (N, C, H, W) → (N, H, W, C)  for nn.LayerNorm
        return super().forward(x.permute(0, 2, 3, 1))        \
                     .permute(0, 3, 1, 2)

def fill_invalid_waypoints(
    agent_traj: torch.Tensor,         # (B, A, T, 3)
    agent_traj_label: torch.Tensor,   # [B, A, T]  (bool)
    agent_labels: torch.Tensor        # [B, A]     (bool)
) -> torch.Tensor:
    """
    Linear interpolate gaps; linearly extrapolate tail.
    First waypoint of each existing agent is valid.
    """
    B, A, T, C = agent_traj.shape
    device = agent_traj.device

    traj_flat   = agent_traj.reshape(B * A, T, C)          # (N, T, 3)
    label_flat  = agent_traj_label.reshape(B * A, T).to(torch.bool)       # (N, T)
    keep_mask   = agent_labels.reshape(B * A).to(torch.bool)              # [N]
    N           = int(keep_mask.sum())

    if N == 0:
        return agent_traj

    traj = traj_flat[keep_mask]                         # (N, T, 3)
    mask = label_flat[keep_mask]                        # (N, T)

    idx   = torch.arange(T, device=device)              # [T]
    idx_b = idx.unsqueeze(0).expand(N, -1)              # (N, T)

    # prev valid index
    prev_init = torch.where(mask, idx_b, torch.zeros_like(idx_b))
    prev_idx  = torch.cummax(prev_init, 1)[0]           # (N, T)

    # next valid index (T sentinel)
    nxt_init  = torch.where(mask, idx_b, torch.full_like(idx_b, T))
    nxt_idx   = torch.flip(torch.cummin(torch.flip(nxt_init, [1]), 1)[0], [1])  # (N, T)

    # ----- interpolation part -----
    mid_mask   = (~mask) & (nxt_idx < T)                # gaps inside valid range
    batch_ids  = torch.arange(N, device=device).unsqueeze(1)

    prev_vals  = traj[batch_ids, prev_idx]              # (N, T, 3)
    next_vals  = traj[batch_ids, nxt_idx.clamp(max=T-1)]

    denom      = (nxt_idx - prev_idx).clamp(min=1).unsqueeze(-1).float()
    alpha      = (idx_b - prev_idx).unsqueeze(-1).float() / denom
    interp     = prev_vals + alpha * (next_vals - prev_vals)

    traj[mid_mask] = interp[mid_mask]

    # ----- tail extrapolation part -----
    last_idx   = prev_idx[:, -1]                        # [N]
    tail_mask  = idx_b > last_idx.unsqueeze(1)          # (N, T)
    if tail_mask.any():
        last_idx_clamped = last_idx.clamp(min=1)                # prevent -1
        gather_idx = (last_idx_clamped - 1).unsqueeze(1)        # (N, 1)
        second_last_idx_gathered = torch.gather(prev_idx, 1, gather_idx).squeeze(1)  # [N]
        second_last_idx = torch.where(last_idx > 0, second_last_idx_gathered, last_idx) # [N]

        # final result
        last_vals = torch.gather(traj, 1, last_idx[:,None,None].expand(-1, -1, C)).squeeze(1) # (N, 3)
        second_vals = torch.gather(traj, 1, second_last_idx[:,None,None].expand(-1, -1, C)).squeeze(1) # (N, 3)
        slope       = (last_vals - second_vals)         # (N, 3)

        offset      = (idx_b - last_idx.unsqueeze(1)).unsqueeze(-1).float()  # (N, T, 1)
        extrap      = last_vals.unsqueeze(1) + slope.unsqueeze(1) * offset   # (N, T, 3)
        traj[tail_mask] = extrap[tail_mask]

    traj_flat[keep_mask] = traj
    filled_traj = traj_flat.reshape(B, A, T, C)

    return filled_traj

def pair_NC_loss_GT_calculate(
                       plan_trajs: torch.Tensor,
                       agent_trajs: torch.Tensor,
                       agent_motion_mask: torch.Tensor,
                       agent_states: torch.Tensor,
                       agent_valid_mask: torch.Tensor,
                       ins_box_margin: Tuple[float,float]=(1.4, 1.9), # (width, length)
                       ego_box_margin: Tuple[float,float]=(1.0, 1.2),
                       margin_type: str = 'distance',
                       ) -> tuple[torch.Tensor]:
    """Pair-wise no-collision GT from box overlap between the plan and each agent."""
    device = plan_trajs.device
    B, K, T, _ = plan_trajs.shape

    agent_traj   = agent_trajs[:, :, None, 1:].repeat(1,1,K,1,1)
    agent_mask = agent_motion_mask[:, :, None, 1:].repeat(1,1,K,1).bool()
    agent_states = agent_states[:, :, None, None].repeat(1,1,K,T,1)

    P = agent_traj.shape[1]

    vm = agent_valid_mask.bool()  # (B, P)
    vm = vm.unsqueeze(-1).expand(-1, -1, K)  # (B, P, K)

    if not vm.any():
        empty_collide_mask = torch.zeros(B, P, K, dtype=torch.float, device=device)
        empty_corners = torch.zeros(B, P, K, T, 4, 2, dtype=torch.float, device=device)
        empty_collide_time = torch.zeros(B, P, K, T, dtype=torch.bool, device=device)
        empty_agent_mask = torch.zeros(B, P, K, T, dtype=torch.bool, device=device)
        empty_ego_corners = torch.zeros(B, K, T, 4, 2, dtype=torch.float, device=device)
        return (empty_collide_mask, empty_ego_corners, empty_corners, empty_collide_time, empty_agent_mask)

    agent_motion_box = torch.cat([agent_traj[...,:7], agent_states[..., 3:5]],dim=-1)
    if margin_type == 'fixed':
        agent_motion_box[..., 3] *= ins_box_margin[1]
        agent_motion_box[..., 4] *= ins_box_margin[0]
    elif margin_type == 'distance':
        ins_box_margin_ratio = (torch.norm(agent_motion_box[..., :2], dim=-1) / 10.0).clamp(min=1.0)
        ins_box_margin_width = ins_box_margin_ratio.clamp(max=ins_box_margin[1])
        ins_box_margin_length = ins_box_margin_ratio.clamp(max=ins_box_margin[0])
        agent_motion_box[..., 3] *= ins_box_margin_width
        agent_motion_box[..., 4] *= ins_box_margin_length

    x, y, yaw, w, h = agent_motion_box[..., 0], agent_motion_box[..., 1], -agent_motion_box[..., 2], agent_motion_box[..., 3], agent_motion_box[..., 4]
    dx = w / 2
    dy = h / 2
    corners_local = torch.stack([torch.stack([dx, dy], dim=-1),torch.stack([dx, -dy], dim=-1), torch.stack([-dx, -dy], dim=-1), torch.stack([-dx, dy], dim=-1)], dim=-2)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    rot_mat = torch.stack([torch.stack([cos_yaw, -sin_yaw], dim=-1),torch.stack([sin_yaw,  cos_yaw], dim=-1)], dim=-2)  # shape : [4, 11, 8, 2, 2]
    rotated_corners = torch.matmul(corners_local, rot_mat)
    corners_world = rotated_corners + torch.stack([x, y], dim=-1)[..., None, :]

    # planning boxes
    half_width = 1.1485 * ego_box_margin[0]
    front_length=4.049 * ego_box_margin[1]
    rear_length=1.127 * ego_box_margin[1]
    ego_corners = torch.tensor([[half_width, front_length], [-half_width, front_length], [-half_width, -rear_length], [half_width, -rear_length]]).to(agent_states)
    ego_corners = torch.tensor([[front_length, half_width], [front_length, -half_width], [-rear_length, -half_width],[-rear_length, half_width] ]).to(agent_states)
    ego_corners = ego_corners[None, None, None].repeat(B, K, T, 1, 1)
    ego_yaw = -plan_trajs[..., 2]
    cos_yaw = torch.cos(ego_yaw)
    sin_yaw = torch.sin(ego_yaw)
    rot_mat = torch.stack([torch.stack([cos_yaw, -sin_yaw], dim=-1),torch.stack([sin_yaw,  cos_yaw], dim=-1)], dim=-2)
    rotated_corners = torch.matmul(ego_corners, rot_mat)
    ego_corners_world = rotated_corners + plan_trajs[..., None, :2]

    device = corners_world.device
    B, A, K, T = corners_world.shape[:4]
    ego_rep = ego_corners_world.unsqueeze(1).expand(-1, A, -1, -1, -1, -1)      # (L, A, K, T, 4, 2)
    agent_corners = corners_world                                              # (L, A, K, T, 4, 2)
    edges_ego   = ego_rep[..., [1, 2], :] - ego_rep[..., [0, 1], :]            # (L, A, K, T, 2, 2)
    edges_agent = agent_corners[..., [1, 2], :] - agent_corners[..., [0, 1], :]
    perp = lambda v: torch.stack([-v[..., 1], v[..., 0]], dim=-1)
    axes_ego   = torch.stack([perp(edges_ego[..., 0, :]), perp(edges_ego[..., 1, :])], dim=-2)   # (L, A, K, T, 2, 2)
    axes_agent = torch.stack([perp(edges_agent[..., 0, :]), perp(edges_agent[..., 1, :])], dim=-2)
    all_axes = torch.cat([axes_ego, axes_agent], dim=-2)                     # (L, A, K, T, 4, 2)
    axes_norm = all_axes / (torch.norm(all_axes, dim=-1, keepdim=True) + 1e-8)
    proj_ego = (ego_rep.unsqueeze(-3) * axes_norm.unsqueeze(-2)).sum(dim=-1)     # (L, A, K, T, 4, 4)
    proj_agent = (agent_corners.unsqueeze(-3) * axes_norm.unsqueeze(-2)).sum(dim=-1)
    ego_min, ego_max = proj_ego.min(dim=-1).values, proj_ego.max(dim=-1).values  # (L, A, K, T, 4)
    ag_min, ag_max   = proj_agent.min(dim=-1).values, proj_agent.max(dim=-1).values
    overlap_each_axis = (ego_max >= ag_min) & (ag_max >= ego_min)  # (L, A, K, T, 4)
    collide_mask_time = overlap_each_axis.all(dim=-1)                   # (L, A, K, T)
    collide_mask_time[~agent_mask] = 0
    collide_mask  = collide_mask_time.any(dim=-1)                        # (B, A, K)

    return (collide_mask, ego_corners_world, corners_world, collide_mask_time, agent_mask)

def get_seg_prob(
    bev_segmap: torch.Tensor,                         # (B,C,H,W)
    plan_traj: torch.Tensor,                          # (B,T,3) or (B,K,T,3) with (x[m],y[m],yaw[rad]) - rear axle coordinates
    reduction: str = 'min',
    interpolation: bool = True,
    class_idx: int = 1,
    num_points: int = 9,
    vehicle_parameters: VehicleParameters = get_pacifica_parameters(),
    x_scale: float = 8.0,                             # px/m for forward (x) - default 8.0 to match safedrive_loss
    y_scale: float = 8.0,                             # px/m for lateral (y) - default 8.0 to match safedrive_loss
    y_min: float = -32.0,                             # meters (left boundary)
    bbox_margin: Tuple[float,float] = (1.0, 1.0),
) -> torch.Tensor:
    """
    For each waypoint (rear axle coordinates), sample the target class prob inside the rotated vehicle box (4 or 9 points)
    and reduce (min) over the sampled points. Returns per-waypoint probability.

    Note: plan_traj waypoints are at rear axle, not box center. Box corners are defined relative to rear axle.

    Returns:
        (B,T) if plan_traj is (B,T,3); or (B,K,T) if (B,K,T,3)
    """
    assert reduction in ['min', 'prod']
    assert num_points in [4, 9]
    assert bev_segmap.dim() == 4, "bev_segmap must be (B,C,H,W)"

    B, C, H, W = bev_segmap.shape
    seg = bev_segmap[:, class_idx:class_idx+1]  # (B,1,H,W)

    # --- support (B,T,3) and (B,K,T,3)
    if plan_traj.dim() == 3:
        B2, T, three = plan_traj.shape
        assert B2 == B and three == 3, "plan_traj must be (B,T,3)"
        BK = B
        K  = None
        traj = plan_traj
        flat_traj = traj.reshape(B, T, 3)
    elif plan_traj.dim() == 4:
        B2, K, T, three = plan_traj.shape
        assert B2 == B and three == 3, "plan_traj must be (B,K,T,3)"
        BK = B * K
        traj = plan_traj
        flat_traj = traj.reshape(BK, T, 3)
        # replicate seg to match BK
        seg = seg.repeat_interleave(K, dim=0)  # (B*K,1,H,W)
    else:
        raise ValueError("plan_traj must be (B,T,3) or (B,K,T,3)")

    device = bev_segmap.device
    dtype  = bev_segmap.dtype

    # --- vehicle size in meters (rear axle relative)
    front_length = float(vehicle_parameters.front_length)  # 4.049
    rear_length = float(vehicle_parameters.rear_length)     # 1.127
    half_width = float(vehicle_parameters.width) / 2.0      # 1.1485

    half_width *= bbox_margin[0]
    front_length *= bbox_margin[1]
    rear_length *= bbox_margin[1]

    # --- split traj (rear axle coordinates)
    yaw  = -flat_traj[..., 2]  # (BK,T) radians

    # --- Define box corners in meters relative to rear axle (matching safedrive_loss.py)
    # Corners: [front_left, front_right, rear_right, rear_left]
    # Format: (x, y) where x is forward, y is lateral
    if num_points == 4:
        ego_corners_m = torch.tensor([
            [front_length, half_width],    # front left
            [front_length, -half_width],   # front right
            [-rear_length, -half_width],   # rear right
            [-rear_length, half_width]     # rear left
        ], device=device, dtype=dtype)  # (4, 2)
    elif num_points == 9:
        # 3x3 grid: front/rear × left/center/right
        xs = torch.tensor([-rear_length, (front_length + rear_length) / 2 - rear_length, front_length], device=device, dtype=dtype)
        ys = torch.tensor([-half_width, 0.0, half_width], device=device, dtype=dtype)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
        ego_corners_m = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (9, 2)
    else:
        raise ValueError(f"num_points must be 4 or 9, got {num_points}")

    # Expand corners to match batch dimensions: (BK, T, N, 2)
    ego_corners_m = ego_corners_m[None, None, :, :].expand(BK, T, -1, -1)  # (BK, T, N, 2)

    # --- Rotate corners by yaw
    cos_yaw = torch.cos(yaw)  # (BK, T)
    sin_yaw = torch.sin(yaw)  # (BK, T)
    rot_mat = torch.stack([
        torch.stack([cos_yaw, -sin_yaw], dim=-1),  # (BK, T, 2)
        torch.stack([sin_yaw,  cos_yaw], dim=-1)   # (BK, T, 2)
    ], dim=-2)  # (BK, T, 2, 2)

    # Rotate: (BK, T, N, 2) @ (BK, T, 2, 2) -> (BK, T, N, 2)
    rotated_corners_m = torch.matmul(ego_corners_m, rot_mat)

    # --- Translate to world coordinates (rear axle position)
    rear_axle_xy = flat_traj[..., :2].unsqueeze(-2)  # (BK, T, 1, 2)
    ego_corners_world_m = rotated_corners_m + rear_axle_xy  # (BK, T, N, 2)

    # --- Convert to pixel coordinates
    # x: * x_scale, y: (y + 32) * y_scale (matching safedrive_loss.py)
    ego_corners_px = ego_corners_world_m.clone()
    ego_corners_px[..., 0] = ego_corners_px[..., 0] * x_scale  # x (forward)
    ego_corners_px[..., 1] = (ego_corners_px[..., 1] - y_min) * y_scale  # y (lateral)

    # --- grid_sample needs normalized coords in [-1,1]
    # grid[..., 0] = x_w (cols, y_px) normalized by W
    # grid[..., 1] = y_h (rows, x_px) normalized by H
    xw = ego_corners_px[..., 1]  # y_px -> width axis
    yh = ego_corners_px[..., 0]  # x_px -> height axis

    x_norm = (xw / max(W - 1, 1) - 0.5) * 2.0   # (BK,T,N)
    y_norm = (yh / max(H - 1, 1) - 0.5) * 2.0   # (BK,T,N)

    grid = torch.stack([x_norm, y_norm], dim=-1)              # (BK,T,N,2)
    grid = grid.view(BK, T * num_points, 1, 2).to(dtype)      # (BK,T*N,1,2)

    if interpolation:
        # bilinear sampling; border padding
        sampled = F.grid_sample(
            seg, grid, mode='bilinear', padding_mode='border', align_corners=True
        )  # (BK,1,T*N,1)
        sampled = sampled.view(BK, T, num_points)  # (BK,T,N)
    else:
        # nearest neighbor: round and index
        yh_nn = yh.round().clamp_(0, H - 1).long()  # (BK,T,N)
        xw_nn = xw.round().clamp_(0, W - 1).long()  # (BK,T,N)
        # seg: (BK,1,H,W) -> squeeze channel
        seg2 = seg.squeeze(1)                       # (BK,H,W)
        # gather per BK
        idx_bk = torch.arange(seg2.shape[0], device=device).view(-1, 1, 1)  # (BK,1,1)
        sampled = seg2[idx_bk, yh_nn, xw_nn]        # (BK,T,N)

    # reduction per waypoint over N points
    if reduction == 'min':
        probs_wp = sampled.min(dim=-1).values           # (BK,T)
    elif reduction == 'prod':
        probs_wp = sampled.prod(dim=-1)           # (BK,T)

    # reshape back
    if plan_traj.dim() == 3:
        result = probs_wp.view(B, T)                  # (B,T)
    else:
        result = probs_wp.view(B, K, T)               # (B,K,T)

    return result

def parse_agent_states(agent_states: torch.Tensor) -> torch.Tensor:
    """
    Parse agent states from GRAD prediction format to standard bbox format.

    Input format: [x, y, sin, cos, w, h] or [x, y, sin, cos, w, h, vx, vy]
    Output format: [x, y, z, length, width, height, yaw]

    :param agent_states: (B, N, 6) or (B, N, 8) - GRAD prediction format
    :return: (B, N, 7) - standard bounding box format
    """
    if agent_states.size == 0:
        return torch.zeros((0, 0, 7), dtype=torch.float32)

    B, N = agent_states.shape[:2]

    # Extract components
    x = agent_states[..., 0]
    y = agent_states[..., 1]
    sin_yaw = agent_states[..., 2]
    cos_yaw = agent_states[..., 3]
    w = agent_states[..., 4]  # width (length in vehicle frame)
    h = agent_states[..., 5]  # height (width in vehicle frame)
    if agent_states.shape[-1] == 8:
        vel = agent_states[..., 6:8]
    else:
        vel = torch.zeros(B, N, 2, dtype=torch.float32, device=agent_states.device)

    # Compute yaw from sin/cos
    yaw = torch.arctan2(sin_yaw, cos_yaw)

    # Default values for z and height
    z = torch.zeros(B, N, dtype=torch.float32, device=agent_states.device)
    height = torch.ones(B, N, dtype=torch.float32, device=agent_states.device) * 1.0  # Default vehicle height

    # Construct output in [x, y, z, length, width, height, heading] order
    output = torch.stack([x, y, z, w, h, height, yaw, vel[..., 0], vel[..., 1]], dim=-1)

    return output

def calc_const_vel_motion_traj(agent_states: torch.Tensor, num_time_steps: int = 8) -> torch.Tensor:
    time_steps = (torch.arange(num_time_steps, dtype=torch.float32, device=agent_states.device)[None,None,:,None] + 1.0) / 2

    velocity = agent_states[..., 7:9] # (B, N, 2)
    current_pos = agent_states[..., :2].unsqueeze(2)
    future_xy = current_pos + velocity.unsqueeze(2) * time_steps

    agent_trajs_full = agent_states.unsqueeze(2).repeat(1, 1, num_time_steps + 1, 1)
    agent_trajs_full[..., 1:, :2] = future_xy

    return agent_trajs_full


def reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduce a loss tensor. reduction is one of "none", "mean", "sum"."""
    reduction_enum = F._Reduction.get_enum(reduction)
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply an element-wise weight, then reduce."""
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        return reduce_loss(loss, reduction)

    if reduction == "mean":
        # eps keeps avg_factor == 0 (every label ignored) from dividing by zero
        eps = torch.finfo(torch.float32).eps
        return loss.sum() / (avg_factor + eps)
    if reduction != "none":
        raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def py_sigmoid_focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25,
                          reduction="mean", avg_factor=None):
    """Focal loss (https://arxiv.org/abs/1708.02002), pure PyTorch.

    :param pred: predictions, (N, C) with C classes
    :param target: learning target of the prediction
    :param weight: sample-wise loss weight
    :param gamma: exponent of the modulating factor
    :param alpha: balancing factor between positives and negatives
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # pt here denotes (1 - pt) in the paper, hence pt.pow(gamma)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # weight is usually (num_priors,), i.e. no class axis
                weight = weight.view(-1, 1)
            else:
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    return weight_reduce_loss(loss, weight, reduction, avg_factor)
