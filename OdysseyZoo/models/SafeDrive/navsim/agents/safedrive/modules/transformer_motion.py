# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from navsim.agents.safedrive.modules.utils import auto_fp16
from mmengine.model import BaseModule
from navsim.agents.safedrive.modules.temporal_self_attention import TemporalSelfAttention
from navsim.agents.safedrive.modules.spatial_cross_attention import MSDeformableAttention3D
from navsim.agents.safedrive.modules.decoder_motion import Traj_Guided_Deform_Attention
from mmcv.cnn.bricks.transformer import build_positional_encoding


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class MotionTransformer(BaseModule):
    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=False,
                 use_can_bus=False,
                 can_bus_norm=False,
                 use_cams_embeds=True,
                 rotate_center=[0, 32],
                 positional_encoding=None,
                 config=None,
                 **kwargs):
        super(MotionTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self._config = config
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, Traj_Guided_Deform_Attention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                bev_embed,
                query,
                output_coord_sigmoid,
                plan_anchor,
                ego_fut_mode,
                reg_branches=None,
                cls_branches=None,
                plan_reg_branch=None,
                query_key_padding_mask=None,
                query_pos=None, # useless
                planning_only=False,
                dn_agent_motion_traj=None, # (B, A, T, 3)
                allow_world_attention=False,
                route_tokens=None,
                route_seg_valid=None,
                **kwargs):
        bs = query.size(0)
        dtype = bev_embed.dtype
        bev_h, bev_w = bev_embed.shape[-2:]
        bev_mask = torch.zeros((bs, bev_h, bev_w), device=bev_embed.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)     # (B, D, H, W)
        bev_embed = bev_embed + bev_pos
        spatial_flatten = torch.tensor([[bev_w, bev_h]], device=bev_embed.device)
        level_start_index = torch.tensor([0], device=bev_embed.device)
        feat_flatten = bev_embed.permute(0,1,3,2).flatten(2).permute(2,0,1)   # (H*W, B, D)

        if reg_branches is not None:
            reference_points = reg_branches[0](query)
            reference_points = reference_points.reshape(*reference_points.shape[:-1], 8, 3)   # -> (..., T, 3)
        else:
            reference_points = plan_anchor
        if planning_only:
            zero = torch.zeros(*query.shape[:2], 1, 3, device=reference_points.device, dtype=reference_points.dtype)
            reference_points_with_zero = torch.cat([zero, reference_points], dim=2)
            reference_points_with_zero[..., 0] -= self._config.grid_config['x'][0]
            reference_points_with_zero[..., 1] -= self._config.grid_config['y'][0]
            reference_points_with_zero[..., 0] = reference_points_with_zero[..., 0] / (self._config.grid_config['x'][1] - self._config.grid_config['x'][0])
            reference_points_with_zero[..., 1] = reference_points_with_zero[..., 1] / (self._config.grid_config['y'][1] - self._config.grid_config['y'][0])

            init_traj_out = reference_points_with_zero
            query = query.permute(1, 0, 2)                        # attention wants (N, B, D)

        elif self._config.AF_topk or self._config.AF_det_score:
            reference_points[:,-1, ..., :2] = plan_anchor[..., :2]
            zero = torch.zeros(1, 1, 1, 1, 3, device=reference_points.device, dtype=reference_points.dtype)
            zero_expanded = zero.expand(*reference_points.shape[:3], 1, 3)
            reference_points_with_zero = torch.cat([zero_expanded, reference_points], dim=3)

            # denoising trajectories replace the agent references, the plan slot is kept
            if dn_agent_motion_traj is not None:
                reference_points_with_zero = reference_points_with_zero.reshape(bs, -1, *reference_points_with_zero.shape[1:])
                reference_points_with_zero[:,:-1,:-1] = dn_agent_motion_traj.reshape(bs, -1, *dn_agent_motion_traj.shape[1:])
                reference_points_with_zero = reference_points_with_zero.reshape(-1, *reference_points_with_zero.shape[2:])

            # the plan is in ego metres, the agents already sit in BEV coords
            reference_points_with_zero[:,-1, ..., 0] -= self._config.grid_config['x'][0]
            reference_points_with_zero[:,-1, ..., 1] -= self._config.grid_config['y'][0]
            reference_points_with_zero[..., 0] = reference_points_with_zero[..., 0] / (self._config.grid_config['x'][1] - self._config.grid_config['x'][0])
            reference_points_with_zero[..., 1] = reference_points_with_zero[..., 1] / (self._config.grid_config['y'][1] - self._config.grid_config['y'][0])

            out_sin = output_coord_sigmoid[..., None, 2:3]
            out_cos = output_coord_sigmoid[..., None, 3:4]
            out_yaw = torch.atan2(out_sin, out_cos)
            init_point = torch.cat([output_coord_sigmoid[..., None, :2], out_yaw], dim=-1)

            reference_points_with_zero[:,:-1] = init_point + reference_points_with_zero[:,:-1]
            # agents and anchors are folded into one token axis for the decoder
            init_traj_out = reference_points_with_zero.flatten(1,2)
            query = query.flatten(1,2).permute(1, 0, 2)           # ((A+1)*K, B, D)
            if query_key_padding_mask is not None:
                query_key_padding_mask = query_key_padding_mask.flatten(1,2).permute(1,0)
        else:
            if plan_anchor is not None:
                reference_points[:,-ego_fut_mode:, ..., :2] = plan_anchor[..., :2]
            zero = torch.zeros(1, 1, 1, 3, device=reference_points.device, dtype=reference_points.dtype)
            zero_expanded = zero.expand(*reference_points.shape[:2], 1, 3)

            reference_points_with_zero = torch.cat([zero_expanded, reference_points], dim=2)
            reference_points_with_zero[:,-ego_fut_mode:, ..., 0] -= self._config.grid_config['x'][0] # traslate only planning query
            reference_points_with_zero[:,-ego_fut_mode:, ..., 1] -= self._config.grid_config['y'][0] # traslate only planning query
            reference_points_with_zero[..., 0] = reference_points_with_zero[..., 0] / (self._config.grid_config['x'][1] - self._config.grid_config['x'][0])
            reference_points_with_zero[..., 1] = reference_points_with_zero[..., 1] / (self._config.grid_config['y'][1] - self._config.grid_config['y'][0])

            out_sin = output_coord_sigmoid[..., None, 2:3]
            out_cos = output_coord_sigmoid[..., None, 3:4]
            out_yaw = torch.atan2(out_sin, out_cos)
            init_point = torch.cat([output_coord_sigmoid[..., None, :2], out_yaw], dim=-1)

            init_traj_out = init_point + reference_points_with_zero
            agent_query = query[:,:-ego_fut_mode,None].repeat(1,1,ego_fut_mode,1) # B, A, N_plan, dim
            plan_query = query[:,None,-ego_fut_mode:]                             # B, 1, N_plan, dim
            query = torch.cat([agent_query, plan_query],dim=1).flatten(1,2)
            query = query.permute(1, 0, 2) # 6656, B, 256

            agent_init_traj = init_traj_out[:,:-ego_fut_mode,None].repeat(1,1,ego_fut_mode,1,1)
            plan_init_traj = init_traj_out[:,None,-ego_fut_mode:]
            init_traj_out = torch.cat([agent_init_traj, plan_init_traj],dim=1).flatten(1,2) # B, 6656, 9(T), 3
        # Which rows of the (N*P, B, D) query the SD route may condition. This function owns the
        # three layouts above, so it is the only place that can name the plan rows: planning_only
        # packs nothing but plan anchors, the other two fold agents first and put the plan last.
        route_plan_slice = slice(None) if planning_only else slice(-ego_fut_mode, None)

        inter_states, intermediate_reference_points = self.decoder(
            query=query, # torch.Size([256, 4, 256])
            key=None,
            value=feat_flatten, # torch.Size([4096, 4, 256])
            query_pos=query_pos, # None
            reference_points=init_traj_out, # torch.Size([4, 256, 9, 3])
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=spatial_flatten, # tensor([[64, 64]], device='cuda:0')
            level_start_index=level_start_index, # tensor([0], device='cuda:0')
            output_coord_sigmoid=output_coord_sigmoid, # None
            ego_fut_mode=ego_fut_mode, # 1
            plan_reg_branch=plan_reg_branch,
            config=self._config,
            query_key_padding_mask=query_key_padding_mask,
            allow_world_attention=allow_world_attention,
            route_tokens=route_tokens,
            route_seg_valid=route_seg_valid,
            route_plan_slice=route_plan_slice,
            **kwargs)

        return bev_embed, inter_states, reference_points, intermediate_reference_points
