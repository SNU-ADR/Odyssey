
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

# imported for the side effect: these register themselves under the names the yaml uses
from navsim.agents.safedrive.modules.temporal_self_attention import TemporalSelfAttention
from navsim.agents.safedrive.modules.spatial_cross_attention import MSDeformableAttention3D
from navsim.agents.safedrive.modules.decoder_detection import CustomMSDeformableAttention
from navsim.agents.safedrive.modules.encoder import BEVFormerEncoder


class PerceptionTransformer_encoder(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 encoder,
                 num_feature_levels=1,
                 num_cams=3,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=False,
                 use_can_bus=False,
                 can_bus_norm=False,
                 use_cams_embeds=True,
                 rotate_center=[0, 32],
                 config=None,
                 **kwargs):
        super(PerceptionTransformer_encoder, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.embed_dims = config.tf_d_model
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.rotate_center = rotate_center

        self.level_embeds = nn.Parameter(torch.Tensor(num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(num_cams, self.embed_dims))

        self.config = config
        self.bev_embedding = nn.Embedding(config.bev_w * config.bev_h, self.embed_dims)

        self.init_weights()

    def init_weights(self):
        """Initialize all weights including submodules like deformable attention."""
        nn.init.normal_(self.level_embeds)
        nn.init.normal_(self.cams_embeds)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                prev_bev=None,
                img_metas=None,
                rel_ego_pose=None,
                lidar_bev=None,
                ):
        """
        Forward pass of BEVFormer encoder only model.
        Args:
            mlvl_feats: List of multi-scale camera features.
            bev_queries: Initial BEV queries of shape (bev_h*bev_w, embed_dim).
            bev_pos: Positional encoding of shape (bs, embed_dim, bev_h, bev_w).
            prev_bev: Previous BEV feature (optional).
            img_metas: Meta information including CAN bus.
        Returns:
            BEV feature of shape (bs, bev_h * bev_w, embed_dim)
        """
        bs = mlvl_feats[0].size(0)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        shift = mlvl_feats[0].new_zeros(bs, 2)
        if prev_bev is not None:
            prev_bev = prev_bev.flatten(2).permute(2, 0, 1)
            if rel_ego_pose is not None:
                shift = rel_ego_pose[:,:2]

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2) # 3, 26, 512, 256
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3) # 3, 512, 26, 256

        # `bev_queries` here is the lidar BEV map (bev_h*bev_w, bs, C): it is not the query but the
        # key/value of 'lidar_cross_attn'. None for a camera-only encoder.
        if bev_queries is not None:
            lidar_feat_flatten = []
            lidar_spatial_shapes = []
            lidar_spatial_shapes.append((bev_h, bev_w))
            lidar_feat_flatten.append(bev_queries.clone())
            # bev_queries.shape : torch.Size([4096, 16, 256])
            lidar_feat_flatten = torch.cat(lidar_feat_flatten, 2) # 2048, 26, 256
            lidar_spatial_shapes = torch.as_tensor(lidar_spatial_shapes, dtype=torch.long, device=bev_pos.device)
            lidar_level_start_index = torch.cat((lidar_spatial_shapes.new_zeros((1,)),lidar_spatial_shapes.prod(1).cumsum(0)[:-1]))
        else:
            lidar_feat_flatten = lidar_spatial_shapes = lidar_level_start_index = None
        # dtype/device follow the lidar map as upstream does; bev_pos stands in when there is none.
        emb_ref = bev_queries if bev_queries is not None else bev_pos
        bev_queries = self.bev_embedding.weight.to(emb_ref)[:,None,:].repeat(1, bs, 1) #expand(*bev_queries.shape)

        bev_embed = self.encoder(
            bev_queries, # torch.Size([2048, 26, 256])
            feat_flatten, # torch.Size([3, 512, 26, 256])
            feat_flatten,
            bev_h=bev_h, # 32
            bev_w=bev_w, # 64
            bev_pos=bev_pos, # torch.Size([2048, 26, 256])
            spatial_shapes=spatial_shapes, # tensor([[16, 32]], device='cuda:0')
            level_start_index=level_start_index, # tensor([0], device='cuda:0')
            prev_bev=prev_bev,
            shift=shift,
            img_metas=img_metas, # img_shape: torch.Size([26, 3, 3, 256, 512]), lidar2img: torch.Size([26, 3, 4, 4])
            lidar_feat_flatten=lidar_feat_flatten,
            lidar_spatial_shapes=lidar_spatial_shapes,
            lidar_level_start_index=lidar_level_start_index,
        )
        bev_feature = bev_embed.permute(0, 2, 1).reshape(bs, self.embed_dims, bev_h, bev_w)
        return bev_feature
