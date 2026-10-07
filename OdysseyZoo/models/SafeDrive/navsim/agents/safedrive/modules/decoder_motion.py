"""Deformable decoder for the joint motion/plan head, adapted from BEVFormer."""
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import copy
import warnings
import torch
import torch.nn as nn
from mmengine.model import xavier_init, constant_init

from mmcv.cnn.bricks.transformer import TransformerLayerSequence
import math
from mmengine.model import BaseModule, ModuleList
from mmengine.config import ConfigDict

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmengine.registry import MODELS
from mmcv.cnn import build_norm_layer
from navsim.agents.sdroute import SDRouteCrossAttention


def build_network(cfg, default_args=None):
    return MODELS.build(cfg, default_args=default_args)

class BaseTransformerLayer_Traj(BaseModule):
    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    'and other FFN related arguments '
                    'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]
        super().__init__(init_cfg)
        self.batch_first = batch_first

        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn') + operation_order.count('traj_self_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), 'The length ' \
                f'of attn_cfg {num_attn} is ' \
                'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'traj_self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_network(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))
        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

@MODELS.register_module()
class Motion_Decoder(TransformerLayerSequence):
    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(Motion_Decoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self,
                query,
                *args,
                reference_points=None,
                key_padding_mask=None,
                output_coord_sigmoid=None,
                ego_fut_mode=1,
                reg_branches=None,
                cls_branches=None,
                plan_reg_branch=None,
                config=None,
                route_tokens=None,
                route_seg_valid=None,
                route_plan_slice=None,
                **kwargs):
        output = query
        intermediate = []
        intermediate_reference_points = []
        B, N_ap, num_pose, num_attr = reference_points.shape
        new_reference_points = reference_points
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points.unsqueeze(2)  # BS NUM_QUERY NUM_LEVEL 2

            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                key_padding_mask=key_padding_mask,
                ego_fut_mode=ego_fut_mode,
                route_tokens=route_tokens,
                route_seg_valid=route_seg_valid,
                route_plan_slice=route_plan_slice,
                **kwargs)

            if reg_branches is not None:
                output_ = output.reshape(-1, ego_fut_mode, *output.shape[1:])

                tmp = reg_branches[lid+1](output_[:-1])
                tmp_plan = plan_reg_branch[lid](output_[-1:])

                tmp = torch.cat([tmp, tmp_plan], dim = 0).reshape(*output.shape[:-1], 8, 3)
                tmp = tmp.permute(1,0,2,3).clone()
                tmp = tmp / torch.tensor([config.grid_config['x'][1] - config.grid_config['x'][0],config.grid_config['y'][1] - config.grid_config['y'][0],1.0], device=tmp.device)
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[:,:,1:, :2] = tmp[..., :2] + reference_points[:,:,1:, :2]
                new_reference_points[...,1:, 2] = tmp[..., 2]

                reference_points = new_reference_points.detach()

            elif plan_reg_branch is not None:
                assert ego_fut_mode == 1
                output_ = output.reshape(-1, ego_fut_mode, *output.shape[1:])

                tmp = plan_reg_branch[lid](output_).reshape(*output.shape[:-1], 8, 3) #
                tmp = tmp.permute(1,0,2,3)   # (B, N*P, 8, 3)

                tmp[..., 0] = tmp[..., 0] / (config.grid_config['x'][1] - config.grid_config['x'][0])
                tmp[..., 1] = tmp[..., 1] / (config.grid_config['y'][1] - config.grid_config['y'][0])
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[:,:,1:, :2] = tmp[..., :2] + reference_points[:,:,1:, :2]
                new_reference_points[...,1:, 2] = tmp[..., 2]

                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output


@MODELS.register_module()
class Traj_Guided_Deform_Attention(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 im2col_step=1,
                 dropout=0.1,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None,
                 num_frame=9,
                 timewise_seperate=False,
                 ):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError('embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_frame = num_frame
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_frame * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points * num_frame)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.timewise_seperate = timewise_seperate
        if self.timewise_seperate:
            self.down_dim_layer = nn.Sequential(
                nn.Linear(embed_dims, embed_dims//2),
                nn.ReLU(),
                nn.Linear(embed_dims//2, embed_dims//2),
            )

            self.output_proj = nn.Linear(embed_dims//2 * 9, embed_dims)
        else:
            self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 1,
            2).repeat(1, self.num_levels, self.num_frame, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        if self.timewise_seperate:
            xavier_init(self.down_dim_layer, distribution='uniform', bias=0.)
        self._is_init = True

    # @deprecated_api_warning({'residual': 'identity'},
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',
                **kwargs):
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2) # torch.Size([26, 30, 256])
            value = value.permute(1, 0, 2) # torch.Size([26, 4096, 256])

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1) # torch.Size([26, 4096, 8, 32])
        if self.timewise_seperate:
            sampling_offsets = self.sampling_offsets(query).view(bs, num_query, self.num_frame, self.num_heads, self.num_levels, self.num_points, 2).flatten(1,2)
            attention_weights = self.attention_weights(query).view(bs, num_query, self.num_frame, self.num_heads, self.num_levels * self.num_points)
            attention_weights = attention_weights.softmax(-1)
            attention_weights = attention_weights.flatten(1,2)

            reference_points = reference_points.permute(0,1,3,2,4).flatten(1,2)
            if reference_points.shape[-1] == 2:
                offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, :, None, :]
            else:
                raise ValueError(
                    'Last dim of reference_points must be'
                    f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        else:
            sampling_offsets = self.sampling_offsets(query).view(bs, num_query, self.num_heads, self.num_levels, self.num_frame, self.num_points, 2) # torch.Size([26, 30, 8, 1, 4, 2])
            attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_levels * self.num_points * self.num_frame)
            attention_weights = attention_weights.softmax(-1)
            attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_frame*self.num_points) # torch.Size([26, 30, 8, 1, 4])
            if reference_points.shape[-1] == 2:
                offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                sampling_locations = reference_points[:, :, None, :, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
                sampling_locations = sampling_locations.flatten(4,5)
            elif reference_points.shape[-1] == 4:
                sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            else:
                raise ValueError(
                    'Last dim of reference_points must be'
                    f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if torch.cuda.is_available() and value.is_cuda:
            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if self.timewise_seperate:
            output = output.reshape(bs, num_query, self.num_frame, -1)
            output = self.down_dim_layer(output)
            output = output.reshape(bs, num_query, -1)
        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity

@MODELS.register_module()
class Motion_Decoder_Layer(BaseTransformerLayer_Traj):
    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 bev_h= 64,
                 bev_w= 32,
                 embed_dims = 256,
                 num_frame= 9,
                 use_sdroute=False,
                 sdroute_num_heads=8,
                 sdroute_dropout=0.0,
                 **kwargs):
        super(Motion_Decoder_Layer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)

        self.embed_dims = embed_dims
        self.num_frame = num_frame

        # SD-route branch, reached by the 'route_attn' name in operation_order. That name is
        # not counted by BaseTransformerLayer_Traj's num_attn (which only sees self_attn /
        # cross_attn / traj_self_attn), so attn_cfgs keeps its usual length, and the module is
        # built here rather than by the mmcv registry. It carries its own norms, so no extra
        # 'norm' entry is needed and self.norms is unchanged.
        self.route_attn = None
        if use_sdroute:
            self.route_attn = SDRouteCrossAttention(
                embed_dims, num_heads=sdroute_num_heads,
                d_ffn=feedforward_channels, dropout=sdroute_dropout,
            )
        self.position_encoder = nn.Sequential(
            nn.Linear(num_frame*3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ego_fut_mode=1,
                time_embed=None,
                route_tokens=None,
                route_seg_valid=None,
                route_plan_slice=None,
                **kwargs):
        # The route tensors are named parameters, not **kwargs, on purpose: kwargs is forwarded
        # verbatim into self.attentions[...] below, so anything left in it also reaches the
        # deformable attention and mmcv's MultiheadAttention.
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn('Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, 'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                ref_points = kwargs['reference_points']

                query = query.reshape(-1, ego_fut_mode, *query.shape[1:])    # (N, P, B, C)
                ref_points = ref_points.reshape(*ref_points.shape[:-3], -1)  # (B, N*P, 3*T)

                # (B, N*P, 3*T) -> (N*P, B, C)
                query_pos_ = self.position_encoder(ref_points).permute(1,0,2)
                # (N*P, B, C) -> (N, P, B, C)
                query_pos = query_pos_.reshape(*query.shape)

                kwargs['reference_points'] = kwargs['reference_points'][...,:2]
                N, P, B, C = query.shape

                query = query.flatten(1,2)          # (N, P*B, C)
                query_pos = query_pos.flatten(1,2)  # (N, P*B, C)

                temp_key = temp_value = query
                if query_key_padding_mask is not None:
                    # (N*P, B) -> (N, P, B) -> (N, P*B) -> (P*B, N)
                    query_key_padding_mask = query_key_padding_mask.reshape(-1, ego_fut_mode, *query_key_padding_mask.shape[1:]).flatten(1,2).permute(1,0)

                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1

                query = query.reshape(N,P,B,C).flatten(0,1)
                query_pos = query_pos.reshape(N,P,B,C).flatten(0,1)
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                if 'reference_points' in kwargs:
                    kwargs['reference_points'] = kwargs['reference_points'][..., :2]   # (B, N*P, 1, T, 2)

                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)

                attn_index += 1
                identity = query

            elif layer == 'route_attn':
                # query is (N*P, B, C) here; the route module is batch-first. route_plan_slice
                # selects the plan rows -- slice(None) for ProposalNet_traj (every row is a plan
                # anchor) and slice(-ego_fut_mode, None) for SWNet, where the agents occupy the
                # leading rows. No query_pos is added, as in the other planners that share
                # navsim/agents/sdroute.
                if self.route_attn is not None and route_tokens is not None:
                    sl = route_plan_slice if route_plan_slice is not None else slice(None)
                    conditioned = self.route_attn(
                        query[sl].permute(1, 0, 2), route_tokens, route_seg_valid
                    ).permute(1, 0, 2)
                    query = query.clone()
                    query[sl] = conditioned
                    identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
