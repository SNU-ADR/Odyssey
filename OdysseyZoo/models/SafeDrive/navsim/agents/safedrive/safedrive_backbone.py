
"""
Implements the TransFuser vision backbone.
"""

import timm
import torch.utils.checkpoint as cp
from torch.utils import checkpoint
import torch
import torch.nn.functional as F
from torch import nn

from navsim.agents.safedrive.safedrive_config import SafeDrive_Config
import numpy as np
from mmengine.config import ConfigDict

from navsim.agents.safedrive.modules.utils import auto_fp16
from navsim.agents.safedrive.modules.transformer_encoder import PerceptionTransformer_encoder
from typing import Optional
import spconv.pytorch as spconv
from spconv.pytorch import SparseConv3d, SubMConv3d
class LearnedPositionalEncoding3D(nn.Module):
    """Position embedding with learnable embedding weights.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50):
        super(LearnedPositionalEncoding3D, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """

        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0,
                            1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)

        return pos

    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str


class SafeDrive_Backbone(nn.Module):
    """Multi-scale Fusion Transformer for image + LiDAR feature fusion."""

    def __init__(self, config: SafeDrive_Config):
        super().__init__()
        self.config = config
        try:
            self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True)
        except Exception as e:
            if not config.bkb_path:
                hf_id = timm.models.get_pretrained_cfg(config.image_architecture).hf_hub_id
                raise RuntimeError(
                    f"timm could not load pretrained '{config.image_architecture}' weights from the "
                    f"Hugging Face hub ({e!r}). Pre-fetch hub id '{hf_id}' into the HF cache (HF_HOME), "
                    f"or pass ++config.bkb_path=<local weights file> in the agent config's overrides."
                ) from e
            print(f"Failed to load image encoder with error: {e}; loading {config.bkb_path}")
            self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True,
                                                   pretrained_cfg_overlay=dict(file=config.bkb_path))

        start_index = 0
        # Some networks have a stem layer
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1

        self.neck = SECONDFPN(
            in_channels = self.config.neck_in_channels,
            upsample_strides = self.config.neck_upsample_strides,
            out_channels = self.config.neck_out_channels_second_fpn_channels,
        )

        encoder_cfg = ConfigDict(self.config.bevformer_encoder)
        self.transformer = PerceptionTransformer_encoder(encoder=encoder_cfg, config=config)
        self.bev_h = config.bev_h # 64
        self.bev_w = config.bev_w # 32
        self.embed_dims = config.tf_d_model
        self.positional_encoding = LearnedPositionalEncoding3D(
            num_feats=self.embed_dims//2,
            row_num_embed=self.bev_w,
            col_num_embed=self.bev_h,)
        # extra downsample layer
        self.bev_backbone = CustomResNet(
            numC_input=self.config.bev_backbone_input_dim,
            num_channels=self.config.bev_backbone_output_dim
        )
        self.bev_neck = SECONDFPN(
            in_channels = self.config.bev_neck_in_channels,
            upsample_strides = self.config.bev_neck_upsample_strides,
            out_channels = self.config.bev_out_channels_second_fpn_channels,
        )
        self.bev_downsample = nn.Conv2d(sum(self.config.bev_out_channels_second_fpn_channels), 256, 3, 1, 1)

        self.lidar_upsample = None
        if self.bev_h != 64 or self.bev_w!=32:
            stride = self.bev_h // 64
            self.lidar_upsample= nn.ConvTranspose2d(
                    in_channels=256,
                    out_channels=256,
                    kernel_size=stride,
                    stride=stride
                )

        self.use_lidar = config.use_lidar
        encoder_ops = encoder_cfg['transformerlayers']['operation_order']
        assert self.use_lidar or 'lidar_cross_attn' not in encoder_ops, \
            "use_lidar=False but the BEV encoder operation_order still has 'lidar_cross_attn'"
        self.pc_range = torch.tensor(config.pc_range)
        self.voxel_size = torch.tensor(config.voxel_size)
        self.shape = torch.round((self.pc_range[3:] - self.pc_range[:3]) / self.voxel_size)
        self.shape_np = self.shape.numpy().astype(np.int32)
        self.second_layer = SpMiddleResNetFHD(num_input_features=5, ds_factor=8) if self.use_lidar else None

        self.temporal_concat_layer = nn.Conv2d(config.tf_d_model * config.num_input_frames, config.tf_d_model, 3, 1, 1)

        self.large_grid = None

        self.grid_lower_bound = torch.Tensor([0, -32, -5])
        self.grid_interval = torch.Tensor([1, 1, 1])

    def compute_voxel_and_feats(self, lidar_pc_list):
        """Lidar BEV feature (B, 256, 64, 32) via SECOND; None when camera-only."""
        if not self.use_lidar:
            return None, None
        voxels, coors = [], []
        for pc in lidar_pc_list:
            voxel, coor = voxelization(pc, self.pc_range.to(pc.device), self.voxel_size.to(pc.device))
            voxels.append(voxel)
            coors.append(coor)
        B = len(voxels)
        coors_batch = torch.cat([F.pad(c, (1, 0), mode='constant', value=i) for i, c in enumerate(coors)], dim=0)
        voxels_batch = torch.cat(voxels, dim=0)
        return self.second_layer(voxels_batch, coors_batch, B, self.shape_np)

    def _compute_relative_pose(self, p1, p2, normalize=True):
        dx_global = p2[:, 0] - p1[:, 0]
        dy_global = p2[:, 1] - p1[:, 1]
        dtheta = p2[:, 2] - p1[:, 2]

        cos_theta = torch.cos(p1[:, 2])
        sin_theta = torch.sin(p1[:, 2])

        dx_local = cos_theta * dx_global + sin_theta * dy_global
        dy_local = -sin_theta * dx_global + cos_theta * dy_global

        pose = torch.stack([dx_local, dy_local, dtheta], dim=1)
        if normalize:
            pose[:, 0] /= self.config.x_range
            pose[:, 1] /= self.config.y_range
        return pose

    def single_process(self, T, image_features, img_metas, lidar_features, temporal_concat_list, rel_ego_pose_list):
        """Run one frame through the image backbone and neck."""
        rel_ego_pose = None
        for t in range(T):
            img_feat = image_features[:, t]
            new_img_shape = torch.Size([img_metas['img_shape'][i] for i in [0, 2, 3, 4, 5]])
            new_lidar2img = img_metas['lidar2img'][:, t]
            new_img_metas = dict(img_shape=new_img_shape, lidar2img=new_lidar2img)

            lidar_features_ = lidar_features[t] if self.use_lidar else None
            if t + 1 == T:
                lidar_features = lidar_features_
                break
            lidar_feats, voxel_feature = self.compute_voxel_and_feats(lidar_features_)
            if lidar_feats is not None:
                lidar_feats = lidar_feats.permute(0, 1, 3, 2) # -> B, 256, 32(x), 64(y)

            if t != 0:
                rel_ego_pose = self._compute_relative_pose(img_metas['ego_poses'][:, t - 1], img_metas['ego_poses'][:, t])
            rel_ego_pose_to_cur = self._compute_relative_pose(img_metas['ego_poses'][:, t], img_metas['ego_poses'][:, -1], normalize=False)
            rel_ego_pose_list.append(rel_ego_pose_to_cur)

            bev_embed = self.obtain_bev(
                img_feat,
                lidar_feats,
                new_img_metas,
                rel_ego_pose,
            )
            temporal_concat_list.append(bev_embed)

        return img_feat, lidar_features, temporal_concat_list, rel_ego_pose_list, new_img_metas

    def forward(self, image, lidar, img_metas=None):
        image_features, lidar_features = image, lidar
        img_feat = image_features
        new_img_metas = img_metas
        rel_ego_pose = None

        # previous bev feature
        temporal_concat_list = []
        rel_ego_pose_list = []
        B, T, N, C, imH, imW = image_features.shape
        img_feat, lidar_features, temporal_concat_list, rel_ego_pose_list, new_img_metas =\
            self.single_process(T, image_features, img_metas, lidar_features, temporal_concat_list, rel_ego_pose_list)

        # current bev feature
        lidar_feats, voxel_feature = self.compute_voxel_and_feats(lidar_features)
        if lidar_feats is not None:
            lidar_feats = lidar_feats.permute(0, 1, 3, 2)

        bev_embed = self.obtain_bev(img_feat, lidar_feats, new_img_metas, rel_ego_pose)

        temporal_concat_list.append(bev_embed)
        for t in range(len(rel_ego_pose_list)):
            pose_3x3 = self.pose_to_3x3_transform(rel_ego_pose_list[t])
            temporal_concat_list[t] = self.shift_feature(temporal_concat_list[t],pose_3x3)
        temporal_concat = torch.cat(temporal_concat_list, 1)
        bev_embed = self.temporal_concat_layer(temporal_concat)
        bev_embed = self.bev_backbone(bev_embed)
        bev_embed = self.bev_neck(bev_embed)[0]
        bev_embed = self.bev_downsample(bev_embed)

        return bev_embed

    def obtain_bev(self, image_features, lidar_feats, img_metas, rel_ego_pose):
        """Lift the multi-camera features into the BEV grid, warping the previous BEV in."""
        B, N, C, imH, imW = image_features.shape

        image_features = image_features.reshape(B*N, C, imH, imW)
        image_layers = iter(self.image_encoder.items())
        # Stem layer.
        # In some architectures the stem is not a return layer, so we need to skip it.
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)

        image_feats = []
        # Loop through the 4 blocks of the network.
        image_features_1 = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
        image_features_2 = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features_1)
        image_features_3 = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features_2) # image_features_3.shape
        image_features_4 = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features_3)

        image_feats.extend([image_features_1, image_features_2, image_features_3, image_features_4])
        img_feats = [self.neck(image_feats)[0]]

        mlvl_feats = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            mlvl_feats.append(img_feat.view(B, int(BN / B), C, H, W))
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_mask = torch.zeros((bs, self.bev_w, self.bev_h), device=image_features.device).to(dtype) #(bs, H, W)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)  # mmdet LearnedPositionalEncoding
        # the lidar BEV map is only key/value for 'lidar_cross_attn'; None when camera-only
        bev_queries_ = lidar_feats.flatten(2).permute(2,0,1) if lidar_feats is not None else None

        bev_embed = self.transformer(
                mlvl_feats,
                bev_queries_,
                self.bev_w,
                self.bev_h,
                grid_length=(self.config.x_range / self.bev_w,
                             self.config.y_range / self.bev_h),
                bev_pos=bev_pos,
                img_metas=img_metas, # dict_keys(['img_shape', 'lidar2img'])
                prev_bev=None,
                rel_ego_pose=rel_ego_pose,
        )

        return bev_embed

    def pose_to_3x3_transform(self, xy_yaw: torch.Tensor) -> torch.Tensor:
        x = xy_yaw[..., 0]
        y = xy_yaw[..., 1]
        yaw = xy_yaw[..., 2]

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)

        row1 = torch.stack([cos_yaw, -sin_yaw, x], dim=-1)
        row2 = torch.stack([sin_yaw,  cos_yaw, y], dim=-1)
        row3 = torch.stack([zeros,    zeros,   ones], dim=-1)

        transform = torch.stack([row1, row2, row3], dim=-2)  # (..., 3, 3)
        return transform

    def shift_feature(self, bev_features, sensor2keyegos):
        grid = self.gen_grid(bev_features, sensor2keyegos)
        output = F.grid_sample(bev_features, grid.to(bev_features.dtype), align_corners=True)
        return output

    def gen_grid(self, bev_features, sensor2keyegos):
        """Sampling grid that maps the previous ego frame onto the current one."""
        n, c, h, w = bev_features.shape
        if self.large_grid is None:
            xs = torch.linspace(0, w - 1, w, dtype=bev_features.dtype,device=bev_features.device).view(1, w).expand(h, w)
            ys = torch.linspace(0, h - 1, h, dtype=bev_features.dtype,device=bev_features.device).view(h, 1).expand(h, w)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
            self.large_grid = grid
        else:
            grid = self.large_grid
        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.grid_interval[0]
        feat2bev[1, 1] = self.grid_interval[1]
        feat2bev[0, 2] = self.grid_lower_bound[0]
        feat2bev[1, 2] = self.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)
        swap_mat = torch.tensor([
            [0., 1., 0.],
            [1., 0., 0.],
            [0., 0., 1.],
        ], dtype=sensor2keyegos.dtype, device=sensor2keyegos.device)  # (3, 3)

        sensor2keyegos = swap_mat @ sensor2keyegos @ swap_mat

        tf = torch.inverse(feat2bev).matmul(sensor2keyegos).matmul(feat2bev)[:,None,None]
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],dtype=bev_features.dtype,device=bev_features.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0

        return grid

    def forward_layer_block(self, layers, return_layers, features):
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 norm_cfg=dict(type='BN'),
                 ):
        super(BasicBlock, self).__init__()
        norm1 = nn.BatchNorm2d(planes)
        norm2 = nn.BatchNorm2d(planes)

        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, dilation=dilation, bias=False)
        self.norm1 = norm1

        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.norm2 = norm2

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu1(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out_ = out + identity

            return out_

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out_ = self.relu2(out)

        return out_


# CustomFPN From mmdet3d
class CustomFPN(nn.Module):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 out_ids=[],
                 conv_cfg=None,
                 norm_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 act_cfg=None):
        super(CustomFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.out_ids = out_ids
        if end_level == -1:
            self.backbone_end_level = self.num_ins
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.upsample_cfg = upsample_cfg
        self.fp16_enabled = False

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels, 1, 1, 0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

            self.lateral_convs.append(l_conv)
            if i in self.out_ids:
                fpn_conv = nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )

                self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)

    @auto_fp16(apply_to=('inputs', ))
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in self.out_ids]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)

        return outs[0]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 norm_cfg=dict(type='BN')):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        norm1 = nn.BatchNorm2d(planes)
        norm2 = nn.BatchNorm2d(planes)
        norm3 = nn.BatchNorm2d(planes * self.expansion)

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False
        )
        self.norm1 = norm1

        self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.norm2 = norm2

        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.norm3 = norm3

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity

            return out

        out = _inner_forward(x)

        out = self.relu(out)

        return out


class CustomResNet(nn.Module):
    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
    ):
        super(CustomResNet, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    Bottleneck(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    Bottleneck(curr_numC, curr_numC // 4, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

class SECONDFPN(nn.Module):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 final_conv_feature_dim=None,
                 use_conv_for_no_stride=False):
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPN, self).__init__()
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = nn.ConvTranspose2d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i]
                )
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = nn.Conv2d(
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride
                )

            deblock = nn.Sequential(upsample_layer,
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

        if final_conv_feature_dim is not None:
            self.final_feature_dim = final_conv_feature_dim
            self.final_conv = nn.Sequential(
                nn.Conv2d(in_channels=sum(out_channels), out_channels=sum(out_channels) // 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(sum(out_channels) // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=sum(out_channels) // 2, out_channels=final_conv_feature_dim, kernel_size=1, stride=1))
        else:
            self.final_feature_dim = sum(out_channels)
            self.final_conv = None

    @auto_fp16()
    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]

        if self.final_conv is not None:
            out = self.final_conv(out)

        return [out]


def voxelization(points, pc_range, voxel_size):
    keep = (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) & \
        (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) & \
            (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
    points = points[keep, :]
    coords = ((points[:, [2, 1, 0]] - pc_range[[2, 1, 0]]) /  voxel_size[[2, 1, 0]]).to(torch.int64)
    unique_coords, inverse_indices = coords.unique(return_inverse=True, dim=0)

    voxels = scatter_mean(points, inverse_indices, dim=0)
    return voxels, unique_coords

@torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(dim):
            src = src.unsqueeze(0)
    for _ in range(other.dim()-src.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

@torch.jit.script
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

@torch.jit.script
def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count.clamp_(1)
    count = broadcast(count, out, dim)
    if torch.is_floating_point(out):
        out.div_(count)
    else:
        assert 0
    return out


def conv3x3(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """3x3 convolution with padding"""
    return spconv.SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None

        self.conv1 = conv3x3(inplanes, planes, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

class SpMiddleResNetFHD(nn.Module):
    def __init__(
        self, num_input_features=128, norm_cfg=None, name="SpMiddleResNetFHD", **kwargs
    ):
        super(SpMiddleResNetFHD, self).__init__()
        self.name = name

        self.dcn = None
        self.zero_init_residual = False

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        # input: # (1600, 1200, 41)
        self.conv_input = spconv.SparseSequential(
            SubMConv3d(num_input_features, 16, 3, bias=False, indice_key="res0"),
            build_norm_layer(norm_cfg, 16)[1],
            nn.ReLU(inplace=True)
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
            SparseBasicBlock(16, 16, norm_cfg=norm_cfg, indice_key="res0"),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(
                16, 32, 3, 2, padding=1, bias=False
            ),  # [1600, 1200, 41] -> [800, 600, 21]
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
            SparseBasicBlock(32, 32, norm_cfg=norm_cfg, indice_key="res1"),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(
                32, 64, 3, 2, padding=1, bias=False
            ),  # [800, 600, 21] -> [400, 300, 11]
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
            SparseBasicBlock(64, 64, norm_cfg=norm_cfg, indice_key="res2"),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(
                64, 128, 3, 2, padding=[0, 1, 1], bias=False
            ),  # [400, 300, 11] -> [200, 150, 5]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
            SparseBasicBlock(128, 128, norm_cfg=norm_cfg, indice_key="res3"),
        )

        self.extra_conv = spconv.SparseSequential(
            SparseConv3d(
                128, 128, (3, 1, 1), (2, 1, 1), bias=False
            ),  # [200, 150, 5] -> [200, 150, 2]
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size, input_shape):
        # input: # (41, 1600, 1408)
        sparse_shape = np.array(input_shape[::-1]) + [1, 0, 0]

        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, sparse_shape, batch_size)

        x = self.conv_input(ret)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        ret = self.extra_conv(x_conv4)

        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        multi_scale_voxel_features = {
            'conv1': x_conv1,
            'conv2': x_conv2,
            'conv3': x_conv3,
            'conv4': x_conv4,
        }

        return ret, multi_scale_voxel_features

norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2d),
    "BN1d": ("bn1d", nn.BatchNorm1d),
    "GN": ("gn", nn.GroupNorm),
}

def build_norm_layer(cfg, num_features, postfix=""):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if layer_type not in norm_cfg:
        raise KeyError("Unrecognized norm type {}".format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer
