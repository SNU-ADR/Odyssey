# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import torch
import torch.nn as nn
from mmengine.model import xavier_init

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from navsim.agents.safedrive.modules.utils import auto_fp16
from mmengine.model import BaseModule
from navsim.agents.safedrive.modules.temporal_self_attention import TemporalSelfAttention
from navsim.agents.safedrive.modules.spatial_cross_attention import MSDeformableAttention3D
from navsim.agents.safedrive.modules.decoder_detection import CustomMSDeformableAttention
from mmcv.cnn.bricks.transformer import build_positional_encoding


class DetectionTransformer(BaseModule):
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
        super(DetectionTransformer, self).__init__(**kwargs)
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
        self.reference_points = nn.Linear(self.embed_dims, 2)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                bev_embed,
                query,
                reg_branches=None,
                cls_branches=None,
                reference_points = None,
                query_pos = None,
                attn_mask = None,
                **kwargs):
        bs = query.size(0)
        dtype = bev_embed.dtype
        bev_h, bev_w = bev_embed.shape[-2:]
        bev_mask = torch.zeros((bs, bev_h, bev_w), device=bev_embed.device).to(dtype) # torch.Size([26, 64, 64])
        bev_pos = self.positional_encoding(bev_mask).to(dtype) # torch.Size([26, 256, 64, 64]) or torch.Size([26, 256, 32, 64])
        bev_embed = bev_embed + bev_pos
        spatial_flatten = torch.tensor([[bev_w, bev_h]], device=bev_embed.device)
        level_start_index = torch.tensor([0], device=bev_embed.device)
        feat_flatten = bev_embed.permute(0,1,3,2).flatten(2).permute(2,0,1)

        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)

        # attn_masks for dn is used in only self-attn, not cross-attn.
        attn_masks = None
        if attn_mask is not None:
            attn_masks = [attn_mask, None]

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=feat_flatten,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=spatial_flatten,
            level_start_index=level_start_index,
            attn_masks=attn_masks,
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
