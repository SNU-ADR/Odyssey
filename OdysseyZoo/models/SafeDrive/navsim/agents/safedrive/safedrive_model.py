"""SafeDrive model: ProposalNet -> SWNet -> FRNet.

Shape letters used throughout: B batch, L decoder layers, Q instance queries,
K plan anchors, A selected agents, T future poses, D model dim.
"""
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from navsim.agents.safedrive.inference_compat import scatter_valid
from navsim.agents.safedrive.safedrive_config import SafeDrive_Config
from navsim.agents.safedrive.safedrive_utils import (
    safe_log,
    inverse_sigmoid,
    calc_projection_mats,
    _get_clones,
    ConvNeXtV2Block,
    LayerNorm2d,
    fill_invalid_waypoints,
    get_seg_prob,
)
from navsim.agents.safedrive.safedrive_backbone import SafeDrive_Backbone
from navsim.agents.safedrive.safedrive_features import BoundingBox2DIndex

from navsim.agents.safedrive.modules.blocks import linear_relu_ln, gen_sineembed_for_position
from mmengine.config import ConfigDict

from mmcv.cnn import Linear
from navsim.agents.safedrive.modules.transformer_detection import DetectionTransformer
from navsim.agents.safedrive.modules.transformer_motion import MotionTransformer
from navsim.agents.sdroute import SDRouteSegmentEncoder


class SafeDrive_Model(nn.Module):
    """The three stages as one module; see forward() for the order."""

    def __init__(self, config: SafeDrive_Config):
        """Build the three stages: ProposalNet -> SWNet -> FRNet."""
        super().__init__()
        self._config = config

        self._build_proposal_net(config)
        if not config.no_planning:
            self._build_swnet(config)
        self._build_frnet(config)

        # One encoder for both plan decoders: the 24 route tokens are built once per forward and
        # read by every route_attn in ProposalNet_traj and SWNet, so the two stages cannot drift
        # onto different summaries of the same route.
        self._sdroute_encoder = None
        if getattr(config, "use_sdroute", False) and not config.no_planning:
            self._sdroute_encoder = SDRouteSegmentEncoder(config.tf_d_model)

    def _build_proposal_net(self, config: SafeDrive_Config):
        """BEV backbone and heads, plus the instance detection transformer."""
        self.ProposalNet_BEV = SafeDrive_Backbone(config)
        self._query_splits = [1, config.num_bounding_boxes] # The number 1 is meaningless; only for reproducibility
        self.ins_query_emb = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        bev_height_res = config.lidar_resolution_height

        def build_bev_head(out_channels):
            return nn.Sequential(
                nn.Conv2d(config.bev_features_channels, config.bev_features_channels, 1),
                LayerNorm2d(config.bev_features_channels),
                nn.GELU(),
                ConvNeXtV2Block(config.bev_features_channels),
                ConvNeXtV2Block(config.bev_features_channels),
                nn.Conv2d(config.bev_features_channels, out_channels, 1),
                nn.Upsample(
                    size=(bev_height_res, config.lidar_resolution_width),
                    mode="bilinear",
                    align_corners=False,
                ),)

        self._bev_semantic_head_convnext = build_bev_head(config.num_bev_classes)
        if config.future_bev_frames > 0:
            self.future_bev_semantic_head_convnext = build_bev_head(config.num_bev_classes * config.future_bev_frames)

        # ---- detection ----
        decoder_cfg = ConfigDict(config.bevformer_decoder)
        positional_encoding_cfg = ConfigDict(config.positional_encoding)
        self.ProposalNet_instance = DetectionTransformer(decoder=decoder_cfg, positional_encoding=positional_encoding_cfg, config=config)

        # classification branch
        num_pred = self.ProposalNet_instance.decoder.num_layers
        cls_branch = []
        for _ in range(config.num_reg_fcs):
            cls_branch.append(Linear(config.tf_d_model, config.tf_d_model))
            cls_branch.append(nn.LayerNorm(config.tf_d_model))
            cls_branch.append(nn.ReLU(inplace=True))
        det_num_class = 2 if config.det_pedestrian else 1   # vehicles (+ pedestrians)
        cls_branch.append(Linear(config.tf_d_model, det_num_class))
        fc_cls = nn.Sequential(*cls_branch)

        # regression branch
        self.num_agent_attr = BoundingBox2DIndex.size()
        self.num_agent_attr += 1 # decompose_det_pred_yaw
        self.num_agent_attr += 2
        reg_branch = []
        for _ in range(self._config.num_reg_fcs):
            reg_branch.append(Linear(config.tf_d_model, config.tf_d_model))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(config.tf_d_model, self.num_agent_attr))
        reg_branch = nn.Sequential(*reg_branch)

        self.det_cls_branches = _get_clones(fc_cls, num_pred)
        self.det_reg_branches = _get_clones(reg_branch, num_pred)

        self.ins_ref_points = nn.Embedding(config.num_bounding_boxes+1, 2)  # agent_query, (x,y)
        nn.init.uniform_(self.ins_ref_points.weight, 0.05, 0.95)
        self.ins_pos_emb_layer = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_model),
            nn.LayerNorm(config.tf_d_model),
            nn.ReLU(inplace=True),
            nn.Linear(config.tf_d_model, config.tf_d_model),
        )

    def _build_swnet(self, config: SafeDrive_Config):
        """Plan proposal encoder and the joint motion / plan decoder."""
        plan_cls_branch = nn.Sequential(
            *linear_relu_ln(config.tf_d_model, 1, 2),
            nn.Linear(config.tf_d_model, 1),
        )

        plan_reg_branch = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_model),
            nn.ReLU(),
            nn.Linear(config.tf_d_model, config.tf_d_model),
            nn.ReLU(),
            nn.Linear(config.tf_d_model, config.num_pose * 3),
        )

        positional_encoding_cfg = ConfigDict(self._config.positional_encoding)
        SWNet_cfg = ConfigDict(self._config.SWNet_cfg)
        self.SWNet = MotionTransformer(decoder=SWNet_cfg, positional_encoding=positional_encoding_cfg, config=config)
        num_motion_pred = self.SWNet.decoder.num_layers

        swnet_ins_reg_branch = []
        for _ in range(self._config.num_reg_fcs):
            swnet_ins_reg_branch.append(Linear(config.tf_d_model, config.tf_d_model))
            swnet_ins_reg_branch.append(nn.ReLU())
        swnet_ins_reg_branch.append(Linear(config.tf_d_model, self._config.num_pose * 3))
        swnet_ins_reg_branch = nn.Sequential(*swnet_ins_reg_branch)

        self.swnet_ins_reg_branch = _get_clones(swnet_ins_reg_branch, num_motion_pred + 1)
        self.swnet_ins_cls_branch = None

        self.swnet_plan_cls_branch = _get_clones(plan_cls_branch, config.SWNet_num_layers)
        self.swnet_plan_reg_branch = _get_clones(plan_reg_branch, config.SWNet_num_layers)

        plan_anchor = np.load(config.plan_anchor_path)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # 20, num_pose, 2

        plan_dim = config.tf_d_model
        plan_input_dim = 640
        self.plan_anchor_encoder = nn.Sequential(*linear_relu_ln(config.tf_d_model, 1, 1, plan_input_dim),nn.Linear(config.tf_d_model, plan_dim),)
        # status_feature is [driving_command(4), vel(2), accel(2)]; the sdroute variant drops the
        # command, so the encoder narrows to vel+accel. See _proposal_trajectory for the slice.
        status_dim = 2 + 2 if getattr(config, "drop_driving_command", False) else 4 + 2 + 2
        self._status_encoding = nn.Linear(status_dim, plan_dim)
        # ProposalNet trajectory decoder (stage 1); absent when the config has no cfg for it
        self.ProposalNet_traj = None
        self.prop_traj_n_layers = 0
        if config.proposal_traj_cfg is not None:
            self.ProposalNet_traj = MotionTransformer(decoder=config.proposal_traj_cfg, positional_encoding=positional_encoding_cfg, config=config)
            self.prop_traj_n_layers = config.prop_traj_n_layers
            self.proposal_traj_cls_branch = _get_clones(plan_cls_branch, self.prop_traj_n_layers)
            self.proposal_traj_reg_branch = _get_clones(plan_reg_branch, self.prop_traj_n_layers)

    def _build_frnet(self, config: SafeDrive_Config):
        """Safety heads: scene-level, pair-wise NC / Disp, time-wise DAC, and the
        test-time scoring weights."""
        positional_encoding_cfg = ConfigDict(config.positional_encoding)

        self.ins_box_margin = (1.0, 1.0)
        self.ego_box_margin = (1.0, 1.0)
        self.margin_type = 'fixed'

        self.twdac = False
        self.TwDAC_bevseg_pred = False
        self.twdac_bevseg_filter_thr = 0.0
        self.TwDAC_bbox_margin = (1.0, 1.0)

        if not config.no_planning:
            if self._config.scene_level_safety:
                self.scene_level_safety_heads = nn.ModuleDict({
                    'NC': nn.Sequential(
                        nn.Linear(config.tf_d_model, config.tf_d_ffn),
                        nn.ReLU(),
                        nn.Linear(config.tf_d_ffn, 1),
                    ),
                    'DAC': nn.Sequential(
                        nn.Linear(config.tf_d_model, config.tf_d_ffn),
                        nn.ReLU(),
                        nn.Linear(config.tf_d_ffn, 1),
                        ),
                    'TTC': nn.Sequential(
                        nn.Linear(config.tf_d_model, config.tf_d_ffn),
                        nn.ReLU(),
                        nn.Linear(config.tf_d_ffn, 1),
                    ),
                    'EP': nn.Sequential(
                        nn.Linear(config.tf_d_model, config.tf_d_ffn),
                        nn.ReLU(),
                        nn.Linear(config.tf_d_ffn, 1),
                    ),
                })

                if self._config.pred_DDC:
                    self.scene_level_safety_heads['DDC'] = nn.Sequential(
                            nn.Linear(config.tf_d_model, config.tf_d_ffn),
                            nn.ReLU(),
                            nn.Linear(config.tf_d_ffn, 1),
                        )
                if self._config.pred_TLC:
                    self.scene_level_safety_heads['TLC'] = nn.Sequential(
                            nn.Linear(config.tf_d_model, config.tf_d_ffn),
                            nn.ReLU(),
                            nn.Linear(config.tf_d_ffn, 1),
                        )
                if self._config.pred_LK:
                    self.scene_level_safety_heads['LK'] = nn.Sequential(
                            nn.Linear(config.tf_d_model, config.tf_d_ffn),
                            nn.ReLU(),
                            nn.Linear(config.tf_d_ffn, 1),
                        )
                if self._config.pred_pdm_score:
                    self.scene_level_safety_heads['pdm_score'] = nn.Sequential(
                            nn.Linear(config.tf_d_model, config.tf_d_ffn),
                            nn.ReLU(),
                            nn.Linear(config.tf_d_ffn, 1),
                        )

        self.imi_test_weight = config.imi_test_weight
        self.NC_test_weight = config.NC_test_weight
        self.DAC_test_weight = config.DAC_test_weight
        self.EP_test_weight = config.EP_test_weight
        self.TTC_test_weight = config.TTC_test_weight
        self.W_test_weight = config.W_test_weight
        self.C_test_weight = config.C_test_weight
        self.pdm_score_test_weight = config.pdm_score_test_weight
        self.DDC_test_weight = config.DDC_test_weight
        self.TLC_test_weight = config.TLC_test_weight
        self.LK_test_weight = config.LK_test_weight
        self.HC_test_weight = config.HC_test_weight
        self.TwDAC_test_weight = config.TwDAC_test_weight

        self.NC_filter_thr = 0.3
        self.DAC_filter_thr = 0.3

        self.disp_scoring_AV_margin = False

        self.pair_NC_scoring = False
        self.PwNC_test_weight = 1.0

        self.twdac_scoring = False

        fine_safe_indim = config.tf_d_model

        mid_dim = config.tf_d_ffn

        if config.pwnc_check:
            NC_final_dim = config.num_pose

            self.pwnc_heads = nn.Sequential(
                    nn.Linear(fine_safe_indim * 2, mid_dim),
                    nn.LayerNorm(mid_dim, eps=1e-6),
                    nn.ReLU(),
                    nn.Linear(mid_dim, mid_dim),
                    nn.LayerNorm(mid_dim, eps=1e-6),
                    nn.ReLU(),
                    nn.Linear(mid_dim, NC_final_dim),
                )

        if config.pwdisp_check:
            self.disp_dim = 2
            if config.pair_Disp_motion_with_yaw:
                self.disp_dim += 2
            disp_dim = config.num_pose * self.disp_dim

            self.pwdisp_heads = nn.Sequential(
                    nn.Linear(fine_safe_indim * 2, mid_dim),
                    nn.LayerNorm(mid_dim, eps=1e-6),
                    nn.ReLU(),
                    nn.Linear(mid_dim, mid_dim),
                    nn.LayerNorm(mid_dim, eps=1e-6),
                    nn.ReLU(),
                    nn.Linear(mid_dim, disp_dim),
                )

        if config.twdac_check:
            dac_dim = config.num_pose

            self.twdac_heads = nn.Sequential(
                    nn.Linear(fine_safe_indim, mid_dim),
                    nn.LayerNorm(mid_dim, eps=1e-6),
                    nn.ReLU(),
                    nn.Linear(mid_dim, mid_dim),
                    nn.LayerNorm(mid_dim, eps=1e-6),
                    nn.ReLU(),
                    nn.Linear(mid_dim, dac_dim),
                )

            if self._config.twdac_bev_module:
                self.twdac_BEV_decoder = MotionTransformer(decoder=config.twdac_bev_deform_cfg, positional_encoding=positional_encoding_cfg, config=config)
                self.time_wise_DAC_BEV_module_layer = config.time_wise_DAC_BEV_module_layer
                self.twdac_BEV_Module = nn.Sequential(
                    nn.Conv2d(config.bev_features_channels, config.bev_features_channels, 1),
                    LayerNorm2d(config.bev_features_channels),
                    nn.GELU(),
                    ConvNeXtV2Block(config.bev_features_channels),
                    ConvNeXtV2Block(config.bev_features_channels),
                    )

                if -1 > 0 and -1 > 0:
                    twdac_bev_height_res = -1
                    twdac_bev_width_res = -1
                else:
                    twdac_bev_height_res = config.lidar_resolution_height
                    twdac_bev_width_res = config.lidar_resolution_width

                self.twdac_BEV_Seg_Head = nn.Sequential(
                    nn.Conv2d(config.bev_features_channels, 4, 1),
                    nn.Upsample(
                        size=(twdac_bev_height_res, twdac_bev_width_res),
                        mode="bilinear",
                        align_corners=False,
                    ),
                    )

        self.no_EP_TC_sum_scoring = False
        self.scoring_test = False

    # DN detr utils
    def _extract_gt_for_dn_2d(self, targets: Dict[str, torch.Tensor]):
        """Collect the GT boxes the denoising queries are built from."""
        states = targets['agent_states']                # (B,Nmax,7) [x,y,yaw,L,W,vx,vy]
        motion_traj = targets['motion_traj'][...,:3]
        motion_mask = targets['motion_mask']
        labels = targets['agent_labels'].long()         # (B,Nmax) 0/1/-1
        device = states.device
        B, Nmax, _ = states.shape

        x   = states[..., 0]
        y   = states[..., 1]
        yaw = states[..., 2]
        L   = states[..., 3]
        W   = states[..., 4]
        velx   = states[..., 5]
        vely   = states[..., 6]

        rad_to_ego = torch.arctan2(y, x)
        in_latent_rad_thresh = torch.logical_and(-self._config.latent_rad_thresh <= rad_to_ego, rad_to_ego <= self._config.latent_rad_thresh,)
        valid = torch.logical_and(in_latent_rad_thresh, labels != -1)
        labels = torch.where(in_latent_rad_thresh, labels, torch.full_like(labels, -1))

        # denoising queries come from the classes the head detects (labels are -1 outside the cone)
        valid = ((labels >= 0) if self._config.det_pedestrian else (labels == 0)) & (L > 0) & (W > 0)

        boxes_abs = torch.stack([x, y, yaw, L, W, velx, vely], dim=-1)   # (B,Nmax,5)

        x_min, x_max = map(float, self._config.grid_config['x'][:2])
        y_min, y_max = map(float, self._config.grid_config['y'][:2])
        x_rng = (x_max - x_min)
        y_rng = (y_max - y_min)

        cx      = (x - x_min) / x_rng
        cy      = (y - y_min) / y_rng
        l_norm  = L / x_rng
        w_norm  = W / y_rng
        boxes_norm = torch.stack([cx, cy, l_norm, w_norm], dim=-1).clamp_(0.0, 1.0)  # (cx, cy, L_norm, W_norm)

        trajs = motion_traj[...,:3]
        motion_mask_ = motion_mask.clone().to(torch.bool)
        motion_mask_[:, :, 0] |= valid
        trajs_interpolated = fill_invalid_waypoints(trajs.clone(), motion_mask_, valid)
        trajs_interpolated = trajs_interpolated - trajs_interpolated[...,:1,:].clone()

        boxes_norm_list, labels_list, boxes_abs_list, trajs_list, trajs_preprocessed_list, traj_mask_list, gt_num = [], [], [], [], [], [], []
        # One sync for all per-sample counts, then python-int slices of a stable "valid first" ordering: the
        # same boxes in the same order as per-sample boolean masks, which cost two-plus syncs per sample.
        counts = valid.sum(1).tolist()
        order = torch.argsort((~valid).to(torch.int8), dim=1, stable=True)      # (B, Nmax)
        def _first(x):
            return torch.gather(x, 1, order.view(B, Nmax, *([1] * (x.dim() - 2))).expand_as(x))
        boxes_norm_s, labels_s, boxes_abs_s = _first(boxes_norm), _first(labels), _first(boxes_abs)
        trajs_s, trajs_interp_s, motion_mask_s = _first(trajs), _first(trajs_interpolated), _first(motion_mask)
        for b in range(B):
            n = int(counts[b])
            boxes_norm_list.append(boxes_norm_s[b, :n])   # (cx, cy, l_norm, w_norm)
            labels_list.append(labels_s[b, :n])
            boxes_abs_list.append(boxes_abs_s[b, :n])     # (x, y, L, W, yaw)
            trajs_list.append(trajs_s[b, :n])
            trajs_preprocessed_list.append(trajs_interp_s[b, :n])
            traj_mask_list.append(motion_mask_s[b, :n])
            gt_num.append(n)

        return {
            'boxes_norm_list': boxes_norm_list,   # (cx, cy, L_norm, W_norm)
            'labels_list': labels_list,
            'boxes_abs_list': boxes_abs_list,     # (x, y, L, W, yaw)
            'trajs_list': trajs_list,
            'trajs_preprocessed_list': trajs_preprocessed_list,
            'traj_mask_list': traj_mask_list,
            'gt_num': gt_num,
            'range_xy': (x_min, x_max, y_min, y_max),
        }

    def _prepare_for_dn_2d(self, batch_size: int, reference_points_2d: torch.Tensor, dn_gt: Dict[str, list]):
        """Build noised queries and the attention mask that keeps them apart."""
        device = reference_points_2d.device
        scalar = int(self._config.dn_scalar)

        boxes_list         = dn_gt['boxes_norm_list']   # list of (Ni,4) [cx,cy,L_norm,W_norm]
        labels_list        = dn_gt['labels_list']       # list of (Ni,)
        boxes_abs_list     = dn_gt['boxes_abs_list']    # list of (Ni,5) [x,y,L,W,yaw]
        trajs_list         = dn_gt['trajs_list']        # list of (Ni,9,3)
        trajs_prepro_list  = dn_gt['trajs_preprocessed_list']        # list of (Ni,9,3)
        traj_mask_list     = dn_gt['traj_mask_list']    # list of (Ni,9)
        gt_num             = dn_gt['gt_num']

        if sum(gt_num) == 0:
            return reference_points_2d, None, None, None

        boxes      = torch.cat(boxes_list, dim=0)       # (ΣN, 4) [cx,cy,L_norm,W_norm]
        labels     = torch.cat(labels_list, dim=0)      # (ΣN,)
        boxes_abs  = torch.cat(boxes_abs_list, dim=0)   # (ΣN, 7) [x,y,yaw,L,W,vx,vy]
        trajs      = torch.cat(trajs_list, dim=0)       # (ΣN, 9, 3)
        trajs_preprocessed = torch.cat(trajs_prepro_list, dim=0)       # (ΣN, 9, 3)
        traj_mask  = torch.cat(traj_mask_list, dim=0)   # (ΣN, 9)
        known_boxes_abs_m = boxes_abs.repeat(scalar, 1) # (scalar*ΣN,5)

        batch_idx = torch.cat([
            torch.full((n,), b, device=device, dtype=torch.long)
            for b, n in enumerate(gt_num)
        ], dim=0)  # (ΣN,)

        known_boxes = boxes.repeat(scalar, 1)      # (scalar*ΣN, 4) [cx,cy,L_norm,W_norm]
        known_trajs = trajs.repeat(scalar, 1, 1)      # (scalar*ΣN, 9, 3)
        known_trajs_preprocessed = trajs_preprocessed.repeat(scalar, 1, 1)      # (scalar*ΣN, 9, 3)
        known_traj_mask = traj_mask.repeat(scalar, 1)      # (scalar*ΣN, 9)
        known_labels= labels.repeat(scalar)
        known_bid   = batch_idx.repeat(scalar)
        known_center = known_boxes[:, 0:2].clone()    # (cx, cy)
        L_norm_x = known_boxes[:, 2].clone()          # L normalized by x-range
        W_norm_y = known_boxes[:, 3].clone()          # W normalized by y-range
        known_scale = torch.stack([L_norm_x, W_norm_y], dim=1)  # (·, 2)
        known_scale = torch.sqrt((known_scale**2).sum(-1))

        if self._config.dn_bbox_noise_scale > 0:
            rand = torch.rand_like(known_center) * 2 - 1.0  # [-1,1]
            known_center = (known_center + rand * (known_scale[:,None] * self._config.box_size_nose + self._config.dn_bbox_noise_scale)).clamp_(0.0, 1.0)

        noised_trajs = known_trajs_preprocessed.clone()

        single_pad = int(max(gt_num))
        pad_size   = single_pad * scalar

        padding_bbox = torch.zeros(pad_size, 2, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        padded_reference_points_2d = torch.cat([padding_bbox, reference_points_2d], dim=1)  # (B, pad+Q, 2)
        padded_motion_trajs = torch.zeros(batch_size, pad_size, 9, 3, device=device)

        map_known_indice = torch.empty(0, dtype=torch.long, device=device)
        if len(gt_num):
            per_img_indices = torch.cat([torch.arange(n, device=device) for n in gt_num], dim=0)
            map_known_indice = torch.cat([per_img_indices + single_pad*i for i in range(scalar)], dim=0).long()

        if known_bid.numel() > 0:
            padded_reference_points_2d[(known_bid.long(), map_known_indice)] = known_center
            padded_motion_trajs[(known_bid.long(), map_known_indice)] = noised_trajs

        N_all = pad_size + reference_points_2d.shape[1]
        attn_mask = torch.zeros((N_all, N_all), dtype=torch.bool, device=device)
        attn_mask[pad_size:, :pad_size] = True
        attn_mask[:pad_size, pad_size:] = True
        for i in range(scalar):
            s, e = single_pad*i, single_pad*(i+1)
            attn_mask[s:e, :s] = True
            attn_mask[s:e,  e:pad_size] = True

        mask_dict = {
            'pad_size': pad_size,
            'single_pad': single_pad,
            'known_bid': known_bid,
            'map_known_indice': map_known_indice,
            'known_labels': known_labels,
            'known_boxes': known_boxes,             # (cx, cy, L_norm, W_norm)
            'known_boxes_abs_m' : known_boxes_abs_m,# (x, y, L, W, yaw)
            'known_trajs': known_trajs,
            'known_traj_mask': known_traj_mask,
            'output_known_logits': None,
            'output_known_boxes' : None,
            'output_known_trajs' : None,
            'known_trajs_preprocessed': noised_trajs,
            'known_center': known_center,
            'gt_num': gt_num,
        }
        return padded_reference_points_2d, padded_motion_trajs, attn_mask, mask_dict

    def _legacy_decode_xy_only(self, q_feat, reference, cls_head, reg_head, grid_cfg):
        # reference: (B, T, 2) in [0,1]
        ref_inv = inverse_sigmoid(reference)
        outputs_class = cls_head(q_feat)              # (B, T, 1)
        tmp = reg_head(q_feat).clone()                # (B, T, K)
        tmp[..., 0:2] += ref_inv[..., 0:2]
        tmp[..., 0:2]  = tmp[..., 0:2].sigmoid()
        output_coord_sigmoid = tmp.clone()
        x0, x1, _ = grid_cfg['x']; y0, y1, _ = grid_cfg['y']
        tmp[..., 0:1] = tmp[..., 0:1] * (x1 - x0) + x0
        tmp[..., 1:2] = tmp[..., 1:2] * (y1 - y0) + y0
        return outputs_class, tmp, output_coord_sigmoid # cls(B,T,1), states(B,T,K) with x,y in meters

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass: ProposalNet -> SWNet -> FRNet.

        `stage` holds the tensors handed from one stage to the next.
        """
        output: Dict[str, torch.Tensor] = {}
        stage: Dict[str, torch.Tensor] = {}

        self._encode_sdroute(features, targets, stage)
        self._forward_proposal_net(features, targets, output, stage)
        if not self._config.no_planning:
            self._forward_swnet(output, stage)
            self._forward_frnet(output, stage)
            if not self.training:
                self._score_at_test_time(output, stage)

        output['training'] = torch.tensor(True) if self.training else torch.tensor(False)
        output['visualize'] = torch.tensor(False)
        return output

    def _encode_sdroute(self, features, targets, stage):
        """Encode the SD route into 24 route tokens, once per forward, for both plan decoders.

        The target builder caches the route (only it sees the map), so `targets` is the usual
        source; `features` is checked too for callers that put it there. A missing route raises
        (see the error below for why and how to fix the cache).
        """
        stage['route_tokens'] = None
        stage['route_seg_valid'] = None
        if self._sdroute_encoder is None:
            return

        route = route_mask = None
        for src in (targets, features):
            if src is not None and "route_centerline" in src:
                route, route_mask = src["route_centerline"], src["route_centerline_mask"]
                break
        if route is None:
            raise RuntimeError(
                "use_sdroute=True but no 'route_centerline' in targets/features. The training "
                "cache predates the route key (SafeDrive_TargetBuilder.get_unique_name() is "
                "unchanged, so old caches still look valid): add the route with "
                "STEPS=3 bash cache_safedrive.sh (navsim/planning/script/run_sdroute_backfill.py)."
            )

        bev_dtype = features["status_feature"].dtype
        tokens, seg_valid = self._sdroute_encoder(route.to(bev_dtype), route_mask.bool())
        stage['route_tokens'] = tokens
        stage['route_seg_valid'] = seg_valid

    def _forward_proposal_net(self, features, targets, output, stage):
        """ProposalNet: BEV encoding, object detection, trajectory proposal."""
        self._proposal_bev_encoder(features, output, stage)
        self._proposal_object_detection(features, targets, output, stage)
        if not self._config.no_planning:
            self._proposal_trajectory(features, output, stage)

    def _proposal_bev_encoder(self, features, output, stage):
        """Fuse the camera and lidar features into the BEV query, then read the BEV maps off it."""
        camera_feature: torch.Tensor = features["camera_feature"]
        if camera_feature.dtype == torch.uint8:
            # cache stores uint8 images; a float32 cache in [0, 1] (upstream's format) passes through
            camera_feature = camera_feature.float().div_(255.0)
        # a lidar-built cache may still carry lidar_feature; a camera-only model just ignores it
        lidar_feature = features["lidar_feature"] if self._config.use_lidar else None
        ego_poses = features["ego_poses"]

        # lidar->image projection, recomputed so the cached matrices stay consistent
        l2i, pr, pt = calc_projection_mats(features['matrices'])
        features['lidar2img'] = l2i
        # post_rots[3] / post_trans[4] inside matrices are replaced by the recomputed values
        mats = features['matrices']
        frame_list = mats if isinstance(mats[0], (list, tuple)) else [mats]
        for frame in frame_list:
            frame[3] = pr
            frame[4] = pt

        lidar2img = features['lidar2img']                       # (B, cam, 4, 4)
        img_metas = {
            'img_shape': camera_feature.shape,                  # (B, frame, cam, 3, H, W)
            'lidar2img' : lidar2img,
            'ego_poses' : ego_poses,
        }
        bev_feature = self.ProposalNet_BEV(camera_feature, lidar_feature, img_metas)  # (B, D, H, W)

        output['bev_semantic_map'] = self._bev_semantic_head_convnext(bev_feature)
        if self._config.future_bev_frames > 0:
            fut_bev_map = self.future_bev_semantic_head_convnext(bev_feature)
            output['fut_bev_map'] = fut_bev_map

        stage['bev_query'] = bev_feature

    def _proposal_object_detection(self, features, targets, output, stage):
        """Decode the instance queries against the BEV query; DN queries ride along in training."""
        bev_feature = stage['bev_query']
        batch_size = features["status_feature"].shape[0]

        # learned instance queries and their reference points, one set per sample
        agents_query = self.ins_query_emb.weight[None, ...].repeat(batch_size, 1, 1)   # (B, Q, D)

        dn_enabled = self.training and self._config.dn_detection
        reference_points = self.ins_ref_points.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        if dn_enabled:
            dn_gt = self._extract_gt_for_dn_2d(targets)
            if self._config.dn_detection:
                reference_points, noised_motion_trajs, attn_mask, dn_mask_dict = self._prepare_for_dn_2d(batch_size, reference_points, dn_gt)    # padded)_ref_2d: (B, pad+matching_q, (x,y)
            pad_size = dn_mask_dict['pad_size'] if dn_mask_dict is not None else 0
        else:
            noised_motion_trajs = None
            attn_mask = None
            dn_mask_dict = None
            pad_size = 0

        if pad_size > 0:
            dn_tgt = torch.zeros((batch_size, pad_size, self._config.tf_d_model)).to(agents_query)
            agents_query = torch.cat([dn_tgt, agents_query], dim=1)    # (B, pad+Q, D)

        query_pos = gen_sineembed_for_position(reference_points, hidden_dim=256)    # (B, pad+Q, D)
        query_pos = self.ins_pos_emb_layer(query_pos)

        outputs = self.ProposalNet_instance(bev_feature, agents_query, self.det_reg_branches, self.det_cls_branches, reference_points = reference_points, query_pos = query_pos, attn_mask=attn_mask) # torch.Size([32, 64, 32, 64])

        bev_embed, det_query, init_reference, inter_references = outputs
        det_query = det_query.permute(0, 2, 1, 3)               # (L, B, pad+Q, D)

        dn_outputs_classes = []
        dn_outputs_coords  = []

        # every decoder layer is decoded and supervised
        for lvl in range(det_query.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]   # (B, pad+Q, 2)
            q_feat    = det_query[lvl]                                              # (B, pad+Q, D)

            cls_logits, decoded, output_coord_sigmoid = self._legacy_decode_xy_only(
                q_feat=q_feat,
                reference=reference,
                cls_head=self.det_cls_branches[lvl],
                reg_head=self.det_reg_branches[lvl],
                grid_cfg=self._config.grid_config,
            )

            if pad_size > 0:
                dn_cls = cls_logits[:, :pad_size, :] # (L,B,pad,1)
                dn_box = decoded[:, :pad_size, :]    # (L,B,pad,K)
                dn_outputs_classes.append(dn_cls)       # (B, pad+Q, 1)
                dn_outputs_coords.append(dn_box)

            ins_logits = cls_logits[:, pad_size:, :]
            output[f'ins_labels_{lvl}'] = ins_logits.squeeze(dim=-1) if ins_logits.shape[-1] == 1 else ins_logits  # (B, Q) or (B, Q, 2)
            output[f'ins_states_{lvl}'] = decoded[:, pad_size:, :]                     # (B, Q, attr)

        if pad_size>0:
            all_dn_outputs_classes = torch.stack(dn_outputs_classes)  # (L, B, pad, 1)
            all_dn_outputs_coords = torch.stack(dn_outputs_coords)    # (L, B, pad, attr)
            dn_mask_dict['output_known_logits'] = all_dn_outputs_classes
            dn_mask_dict['output_known_boxes']  = all_dn_outputs_coords
            output['dn_mask_dict'] = dn_mask_dict

            det_query = det_query[:, :, pad_size:]
            output_coord_sigmoid = output_coord_sigmoid[:, pad_size:]

        if self._config.det_coord_detach:
            output_coord_sigmoid = output_coord_sigmoid.detach()

        stage['ins_query'] = det_query
        stage['ins_coords'] = output_coord_sigmoid
        stage['noised_motion_trajs'] = noised_motion_trajs

    def _proposal_trajectory(self, features, output, stage):
        """Encode the trajectory anchors and refine them into the initial plan proposals."""
        bev_feature = stage['bev_query']
        det_query = stage['ins_query']
        status_feature = features["status_feature"]
        batch_size = status_feature.shape[0]

        instance_query = det_query[-1].clone()

        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(batch_size,1,1,1)   # (B, K, 40, 3)

        traj_pos_embed = gen_sineembed_for_position(plan_anchor, hidden_dim=16)
        plan_anchor = plan_anchor[:, :, 4::5]                                  # 10 Hz -> (B, K, T, 3)

        ego_fut_mode = plan_anchor.shape[1]
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        status_feature = status_feature[:,-1] if status_feature.dim() > 2 else status_feature
        if self._config.drop_driving_command:
            # command is the first 4 of [command(4), vel(2), accel(2)] -- see safedrive_features.
            status_feature = status_feature[..., 4:]
        status_emb = self._status_encoding(status_feature)[:,None]
        traj_feature += status_emb
        traj_feature = traj_feature.view(batch_size,ego_fut_mode,-1)           # (B, K, D)
        output['plan_anchor'] = plan_anchor
        if self.ProposalNet_traj is not None:
            prop_plan_query = traj_feature
            motion_outputs = self.ProposalNet_traj(
                bev_embed=bev_feature,
                query=prop_plan_query,
                output_coord_sigmoid=None,
                plan_anchor=plan_anchor,
                plan_anchor_encoder=self.plan_anchor_encoder,
                ego_fut_mode=1,
                reg_branches=None,
                plan_reg_branch=self.proposal_traj_reg_branch,
                planning_only=True,
                status_emb=status_emb,
                route_tokens=stage['route_tokens'],
                route_seg_valid=stage['route_seg_valid'],
            )

            bev_embed, prop_plan_query, init_traj, intermediate_reference_points = motion_outputs
            prop_plan_query = prop_plan_query.permute(0,2,1,3)
            # anchor offsets come back normalised; put them back into metres
            for lvl in range(self.prop_traj_n_layers):
                plan_reg_pred = intermediate_reference_points[lvl][:,:,1:,]
                plan_reg_pred[..., 0:1] = (plan_reg_pred[..., 0:1] * (self._config.grid_config['x'][1] - self._config.grid_config['x'][0]) + self._config.grid_config['x'][0])
                plan_reg_pred[..., 1:2] = (plan_reg_pred[..., 1:2] * (self._config.grid_config['y'][1] - self._config.grid_config['y'][0]) + self._config.grid_config['y'][0])
                plan_cls = self.proposal_traj_cls_branch[lvl](prop_plan_query[lvl]).squeeze(-1)

                output[f"plan_traj_{lvl}"] = plan_reg_pred
                output[f"plan_traj_cls_{lvl}"] = plan_cls

            traj_feature = prop_plan_query[-1].clone()

            # query filtering
            if self._config.proposalnet_2stage:
                if self._config.scene_level_safety:
                    safety_score_pred = {}
                    for k, head in self.scene_level_safety_heads.items():
                        safety_score_pred[k] = head(prop_plan_query[-1]).squeeze(-1)
                    plan_score = self._config.imi_2stage_weight * safe_log(plan_cls.softmax(-1))
                    NC = self._config.NC_2stage_weight * safe_log(safety_score_pred['NC'].sigmoid())
                    DAC = self._config.DAC_2stage_weight * safe_log(safety_score_pred['DAC'].sigmoid())
                    EP = self._config.EP_2stage_weight * safety_score_pred['EP'].sigmoid()
                    TTC = self._config.TTC_2stage_weight * safety_score_pred['TTC'].sigmoid()
                    W = self._config.W_2stage_weight * safe_log((EP + TTC))
                    score = plan_score + NC + DAC + W
                    if self._config.pred_pdm_score:
                        pdm_score = self._config.pdm_score_2stage_weight * safety_score_pred['pdm_score'].sigmoid()
                        score += pdm_score
                    if self._config.pred_DDC:
                        DDC = self._config.DDC_2stage_weight * safety_score_pred['DDC'].sigmoid()
                        score += DDC
                    if self._config.pred_TLC:
                        TLC = self._config.TLC_2stage_weight * safety_score_pred['TLC'].sigmoid()
                        score += TLC
                    if self._config.pred_LK:
                        LK = self._config.LK_2stage_weight * safety_score_pred['LK'].sigmoid()
                        score += LK
                else:
                    score = plan_cls
                # score[score < -1e+10] = score[score>-1e-10].min()

                # B, N = selected_upper_score.shape
                K = self._config.num_proposal_2stage
                topk_idx = score.topk(K, dim=1).indices  # (B, K)

                idx = topk_idx[:, :, None, None].expand(batch_size, K, 8, 3)
                plan_traj = output[f"plan_traj_{self.prop_traj_n_layers-1}"]
                plan_traj = plan_traj.gather(dim=1, index=idx)

                idx_feat = topk_idx[:, :, None].expand(batch_size, K, self._config.tf_d_model)
                traj_feature = traj_feature.gather(dim=1, index=idx_feat)
                output['stage1_score'] = score
                output['proposal_2stage_topidx'] = topk_idx
            else:
                plan_traj = output[f"plan_traj_{self.prop_traj_n_layers-1}"]

            if self._config.stage1_reference_points_detach:
                plan_traj = plan_traj.detach()
        else:
            plan_traj = plan_anchor

        stage.update({
            'instance_query': instance_query,
            'plan_cls': plan_cls,
            'plan_reg_pred': plan_reg_pred,
            'plan_traj': plan_traj,
            'prop_plan_query': prop_plan_query,
            'traj_feature': traj_feature,
        })

    def _forward_swnet(self, output, stage):
        """SWNet: filter the instances, run the joint motion/plan decoder, emit per-layer outputs."""
        bev_feature = stage['bev_query']
        noised_motion_trajs = stage['noised_motion_trajs']
        output_coord_sigmoid = stage['ins_coords']
        n_agents = self._config.num_bounding_boxes + 1
        instance_query = stage['instance_query']
        plan_traj = stage['plan_traj']
        traj_feature = stage['traj_feature']

        device = bev_feature.device
        query_key_padding_mask = None
        # keep only the instances each plan anchor could interact with
        if self._config.AF_det_score and self._config.AF_topk:
            numel_per_group = [instance_query.shape[1]]
            _, N_anchor, _ = traj_feature.shape

            iq_groups, oc_groups, kpm_groups, orig_groups, valid_groups, mt_groups = [], [], [], [], [], []
            start_idx = 0
            for group_idx, numel in enumerate(numel_per_group):
                end_idx = start_idx + numel

                motion_traj_group = None

                instance_query_group = instance_query[:,start_idx:end_idx,:]
                output_coord_sigmoid_group = output_coord_sigmoid[:,start_idx:end_idx,:]
                B, N_agent, F = instance_query_group.shape

                scores = output['ins_labels_3'].sigmoid()                      # final detection layer
                if scores.dim() == 3:
                    scores = scores.max(dim=-1).values                         # any class (vehicle or pedestrian)
                agent_upper_thr = scores > self._config.ins_filter_cls_thr     # (B, Q)

                sel_query, sel_coord, sel_trajs, det_sorted_idx, det_valid_mask = \
                    self.pack_selected(instance_query_group, output_coord_sigmoid_group, agent_upper_thr, motion_traj_group)

                # per anchor, the A nearest surviving agents
                topk_idx, topk_valid = self.select_topk(agent_boxes=sel_coord,plan_traj=plan_traj,valid_agent_mask=det_valid_mask,agent_sigmoid=True,)

                Cc = output_coord_sigmoid_group.size(-1)
                if topk_idx.numel() > 0:
                    idx_exp_F = topk_idx.unsqueeze(-1).expand(-1, -1, -1, F)     # (B, A, K, D)
                    idx_exp_C = topk_idx.unsqueeze(-1).expand(-1, -1, -1, Cc)    # (B, A, K, attr)
                    idx_exp_T = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1+self._config.num_pose, 3)  # (B, A, K, 1+T, 3)

                    sel_query_exp = sel_query.unsqueeze(2).expand(-1, -1, N_anchor, -1)  # (B, Q', K, D)
                    sel_coord_exp = sel_coord.unsqueeze(2).expand(-1, -1, N_anchor, -1)  # (B, Q', K, attr)
                    sel_trajs_exp = sel_trajs.unsqueeze(2).expand(-1, -1, N_anchor, -1, -1) if sel_trajs is not None else None  # (B, Q', K, 1+T, 3)

                    instance_query_group = torch.gather(sel_query_exp, 1, idx_exp_F)           # (B, A, K, D)
                    output_coord_sigmoid_group = torch.gather(sel_coord_exp, 1, idx_exp_C)     # (B, A, K, attr)
                    motion_traj_group = torch.gather(sel_trajs_exp, 1, idx_exp_T) if motion_traj_group is not None else None

                    instance_query_group = instance_query_group.masked_fill(~topk_valid.unsqueeze(-1), 0)
                    output_coord_sigmoid_group = output_coord_sigmoid_group.masked_fill(~topk_valid.unsqueeze(-1), 0)
                    motion_traj_group = motion_traj_group.masked_fill(~topk_valid.unsqueeze(-1).unsqueeze(-1), 0) if motion_traj_group is not None else None
                else:
                    # nothing passed the threshold: keep the shapes, zero the content
                    K_AF = self._config.num_filtering_instance
                    topk_idx = instance_query.new_zeros((B, K_AF, plan_traj.size(1)), dtype=torch.long)   # (B, A, K)
                    topk_valid = instance_query.new_zeros((B, K_AF, plan_traj.size(1)), dtype=torch.bool) # (B, A, K)
                    Cc = output_coord_sigmoid.size(-1)

                    instance_query_group = instance_query.new_zeros((B, K_AF, N_anchor, F))
                    output_coord_sigmoid_group = output_coord_sigmoid.new_zeros((B, K_AF, N_anchor, Cc))
                    motion_traj_group = noised_motion_trajs.new_zeros((B, K_AF, N_anchor, 9, 3)) if motion_traj_group is not None else None

                query_key_padding_mask = (~topk_valid) if topk_valid.numel() > 0 else topk_valid  # (B, A, K)
                query_key_padding_mask = torch.cat([query_key_padding_mask,
                                                    query_key_padding_mask.new_zeros(B, 1, N_anchor)], dim=1)

                # the plan query rides along as the last 'agent' so one decoder handles both
                instance_query_group = torch.cat([instance_query_group, traj_feature[:, None]], dim=1)  # (B, A+1, K, D)

                if det_sorted_idx.numel() > 0 and topk_idx.numel() > 0:
                    det_sorted_idx_exp = det_sorted_idx.unsqueeze(2).expand(-1, -1, N_anchor)  # (B, Q', K)
                    orig_idx = torch.gather(det_sorted_idx_exp, 1, topk_idx)                   # (B, A, K) back to query ids
                    orig_idx = orig_idx.masked_fill(~topk_valid, 0)
                else:
                    orig_idx = instance_query.new_zeros((B, self._config.num_filtering_instance, N_anchor), dtype=torch.long)

                iq_groups.append(instance_query_group)
                oc_groups.append(output_coord_sigmoid_group)
                kpm_groups.append(query_key_padding_mask)
                orig_groups.append(orig_idx)
                valid_groups.append(topk_valid)
                if motion_traj_group is not None:
                    mt_groups.append(motion_traj_group)

                start_idx += numel

            # groups are folded into the batch axis so one decoder call handles them all
            instance_query = torch.stack(iq_groups, dim=1).flatten(0,1)
            output_coord_sigmoid = torch.stack(oc_groups, dim=1).flatten(0,1)
            query_key_padding_mask = torch.stack(kpm_groups, dim=1).flatten(0,1)
            orig_idx = torch.stack(orig_groups, dim=1).flatten(0,1)
            topk_valid = torch.stack(valid_groups, dim=1).flatten(0,1)
            motion_traj = torch.stack(mt_groups, dim=1).flatten(0,1) if len(mt_groups) > 0 else None

        else:
            instance_query = torch.cat([instance_query, traj_feature], dim = 1)
            zeros_tensor = torch.zeros((*traj_feature.shape[:2], self.num_agent_attr), device=device)
            output_coord_sigmoid = torch.cat([output_coord_sigmoid, zeros_tensor], dim=1)
            motion_traj = None

        attn_masks = None

        ego_fut_mode = instance_query.shape[2]

        motion_outputs = self.SWNet(
            bev_embed=bev_feature,
            query=instance_query,
            attn_masks=attn_masks,
            output_coord_sigmoid=output_coord_sigmoid,
            plan_anchor=plan_traj,
            ego_fut_mode=ego_fut_mode,
            reg_branches=self.swnet_ins_reg_branch,
            cls_branches=self.swnet_ins_cls_branch,
            plan_reg_branch=self.swnet_plan_reg_branch,
            query_key_padding_mask=query_key_padding_mask,
            dn_agent_motion_traj=motion_traj,
            route_tokens=stage['route_tokens'],
            route_seg_valid=stage['route_seg_valid'],
        )

        bev_embed, motion_query, init_traj, intermediate_reference_points = motion_outputs
        motion_query = motion_query.permute(0, 2, 1, 3)     # (L, B, (A+1)*K, D)
        L, B, _, C = motion_query.shape

        motion_query = motion_query.reshape(L, B, -1, ego_fut_mode, C)   # (L, B, A+1, K, D)
        # the last slot is the ego plan, the rest are the agents
        if self._config.AF_det_score or self._config.AF_topk:
            agent_query, plan_query = motion_query.split([instance_query.shape[1] - 1, 1], dim=2)
            motion_layer = agent_query.shape[0]+1
        else:
            agent_query, plan_query = motion_query.split([n_agents, 1], dim=2)
            instance_query = instance_query[:,:-ego_fut_mode,None].repeat(1,1,ego_fut_mode,1)
            agent_query = torch.cat([instance_query[None], agent_query], dim=0)
            motion_layer = agent_query.shape[0]
        plan_query = plan_query.squeeze(2)

        plan_reg_pred = plan_traj
        intermediate_reference_points = intermediate_reference_points.reshape(plan_query.shape[0], B, -1, ego_fut_mode, self._config.num_pose+1, 3)  # (L, B, A+1, K, 1+T, 3)
        pred_intermediate_reference_points = intermediate_reference_points[:,:,:-1,:,1:]
        plan_intermediate_reference_points = intermediate_reference_points[:,:,-1,:,1:]

        # plan head, one output per refinement layer, in metres
        for lvl in range(plan_query.shape[0]):
            plan_reg_pred = plan_intermediate_reference_points[lvl]
            plan_reg_pred[..., 0:1] = (plan_reg_pred[..., 0:1] * (self._config.grid_config['x'][1] - self._config.grid_config['x'][0]) + self._config.grid_config['x'][0])
            plan_reg_pred[..., 1:2] = (plan_reg_pred[..., 1:2] * (self._config.grid_config['y'][1] - self._config.grid_config['y'][0]) + self._config.grid_config['y'][0])
            plan_cls = self.swnet_plan_cls_branch[lvl](plan_query[lvl]).squeeze(-1)
            lvl_idx = lvl + self.prop_traj_n_layers

            output[f"plan_traj_{lvl_idx}"] = plan_reg_pred
            output[f"plan_traj_cls_{lvl_idx}"] = plan_cls

        # motion head; layer 0 is the anchor itself, later layers are offsets from the box
        for lvl in range(motion_layer):
            if lvl == 0:
                if self._config.AF_topk or self._config.AF_det_score:
                    motion_reg = init_traj[:, :-1]
                else:
                    motion_reg = init_traj[:, :-ego_fut_mode][:,:,None].repeat(1,1,ego_fut_mode,1,1)
            else:
                motion_reg = pred_intermediate_reference_points[lvl-1]
                if self._config.AF_topk or self._config.AF_det_score:
                    motion_reg[..., :2] -= output_coord_sigmoid[...,None,:2]
                else:
                    motion_reg[..., :2] -= output_coord_sigmoid[:, :-ego_fut_mode][:,:,None,None,:2]
                motion_reg[..., 0:1] = motion_reg[..., 0:1] * (self._config.grid_config['x'][1] - self._config.grid_config['x'][0])
                motion_reg[..., 1:2] = motion_reg[..., 1:2] * (self._config.grid_config['y'][1] - self._config.grid_config['y'][0])

            if self._config.AF_det_score and self._config.AF_topk:
                B_, K_, P_, T_, D_ = motion_reg.shape

                # scatter the A selected agents back to their original query slots
                n_pad_size = N_agent
                motion_reg_full = torch.zeros(B_, n_pad_size, P_, T_, D_,device=motion_reg.device,dtype=motion_reg.dtype,)
                if orig_idx.numel() > 0:
                    src = motion_reg.masked_fill(~topk_valid.unsqueeze(-1).unsqueeze(-1), 0)  # (B, A, K, T, 3)
                    index_expanded = orig_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, T_, D_)  # (B, A, K, T, 3)
                    motion_reg = scatter_valid(motion_reg_full, 1, index_expanded, src, topk_valid[..., None, None])
                else:
                    motion_reg = motion_reg_full

                mask_full = torch.zeros(B_, n_pad_size, P_, dtype=torch.bool, device=device)
                if orig_idx.numel() > 0:
                    mask_full = scatter_valid(mask_full, 1, orig_idx, topk_valid, topk_valid)
                output['agent_valid_mask'] = mask_full

            output[f"agent_motion_traj_{lvl}"] = motion_reg[:,:,None]

            # at test time the scorer wants one mode per agent, the plan's argmax mode
            if not self.training:
                output[f"agent_motion_traj_{lvl}_for_disp_filtering"] = motion_reg[:, :, None]
                B, A, P, T, _ = motion_reg.shape
                mode_idx = plan_cls.argmax(dim=-1)
                mode_idx = mode_idx[...,None,None,None,None].repeat(1,A,1,self._config.trajectory_sampling.num_poses,3)
                motion_reg = torch.gather(motion_reg, 2, mode_idx)
                output[f"agent_motion_traj_{lvl}"] = motion_reg
        output['n_pad_size'] = n_pad_size

        stage.update({
            'B_': B_,
            'P_': P_,
            'motion_query': motion_query,
            'n_pad_size': n_pad_size,
            'orig_idx': orig_idx,
            'plan_cls': plan_cls,
            'plan_query': plan_query,
            'plan_reg_pred': plan_reg_pred,
            'topk_valid': topk_valid,
        })

    def _forward_frnet(self, output, stage):
        """FRNet: scene-level safety plus the pair-wise NC / pair-wise Disp / time-wise DAC heads."""
        bev_feature = stage['bev_query']
        B_ = stage['B_']
        P_ = stage['P_']
        motion_query = stage['motion_query']
        n_pad_size = stage['n_pad_size']
        orig_idx = stage['orig_idx']
        plan_query = stage['plan_query']
        prop_plan_query = stage['prop_plan_query']
        topk_valid = stage['topk_valid']

        if self._config.scene_level_safety:
            # stage 1 scored a subset of the anchors; pad back to K so the layers line up
            if self._config.proposalnet_2stage:
                orig_len = plan_query.size(2)
                pad_len = self._config.num_plan_anchor - orig_len
                plan_query = nn.functional.pad(plan_query, (0, 0, 0, pad_len))
                pad_mask_stage2 = torch.zeros(plan_query.shape[:3], dtype=torch.bool, device=plan_query.device)  # (L, B, K)
                pad_mask_stage2[:, :, :orig_len] = True
                pad_mask_stage1 = torch.ones(self._config.prop_traj_n_layers, *plan_query.shape[1:3], dtype=torch.bool, device=plan_query.device)
                pad_mask_stage2 = torch.cat([pad_mask_stage1, pad_mask_stage2] ,dim = 0)
                output['pad_mask_stage2'] = pad_mask_stage2

            plan_feat = plan_query if self.ProposalNet_traj is None else torch.cat([prop_plan_query, plan_query], dim=0)

            # one scalar per plan anchor per metric: NC, DAC, EP, TTC, ...
            for k, head in self.scene_level_safety_heads.items():
                pred_score = head(plan_feat).squeeze(-1)
                output[k] = pred_score

            plan_query_pNC_expanded = None
            # pair-wise heads read (agent, plan) pairs, so the plan query is broadcast over agents
            if self._config.pwnc_check or self._config.pwdisp_check:
                agent_query_pNC = motion_query[:, :,:-1]    # (L, B, A, K, D)
                plan_query_pNC = motion_query[:, :,-1:]     # (L, B, 1, K, D)
                L, B, _, K, C = plan_query_pNC.shape
                A = agent_query_pNC.size(2)
                plan_query_pNC_expanded = plan_query_pNC.expand(-1, -1, A, -1, -1)

                concat_tensor = torch.cat([agent_query_pNC, plan_query_pNC_expanded], dim=-1)  # (L, B, A, K, 2D)

                if self._config.pwnc_check:
                    pwnc_pred = self.pwnc_heads(concat_tensor).squeeze(4)
                    L = pwnc_pred.size(0)
                    T = pwnc_pred.size(-1)
                    tv_full = topk_valid[None, :, :, :, None]
                    tv_full = tv_full.expand(L, -1, -1, -1, T)
                    src = (pwnc_pred * tv_full).contiguous()
                    idx_exp = orig_idx[None, :, :, :, None]
                    idx_exp = idx_exp.expand(L, -1, -1, -1, T)
                    # back to the full query slots: (L, B, Q, K, T)
                    all_pair_NC_pred = torch.zeros(L, B_, n_pad_size, P_, T,device=pwnc_pred.device, dtype=pwnc_pred.dtype)

                    all_pair_NC_pred = scatter_valid(all_pair_NC_pred, 2, idx_exp, src, tv_full)

                    output['pwnc_pred'] = all_pair_NC_pred

                if self._config.pwdisp_check:
                    T = self._config.num_pose
                    K_ = self._config.num_filtering_instance
                    pwdisp_pred = self.pwdisp_heads(concat_tensor)                                       # (L, B, A, K, T*2)
                    pwdisp_pred = pwdisp_pred.view(motion_query.shape[0], B_, K_, P_, T, self.disp_dim)  # (L, B, A, K, T, 2)

                    if self._config.AF_det_score and self._config.AF_topk:
                        L = pwdisp_pred.size(0)
                        T = pwdisp_pred.size(-2)
                        tv_full = topk_valid[None, :, :, :, None, None]                          # (1, B, A, K, 1, 1)
                        tv_full = tv_full.expand(L, -1, -1, -1, T, self.disp_dim)               # (L, B, A, K, T, 2)
                        src = (pwdisp_pred * tv_full).contiguous()                              # (L, B, A, K, T, 2)
                        idx_exp = orig_idx[None, :, :, :, None, None]                           # (1, B, A, K, 1, 1)
                        idx_exp = idx_exp.expand(L, -1, -1, -1, T, self.disp_dim)               # (L, B, A, K, T, 2)
                        all_pair_Disp_pred = torch.zeros(L, B_, n_pad_size, P_, T, self.disp_dim,device=pwdisp_pred.device, dtype=pwdisp_pred.dtype)
                        all_pair_Disp_pred = scatter_valid(all_pair_Disp_pred, 2, idx_exp, src, tv_full)

                        output['pwdisp_pred'] = all_pair_Disp_pred
                        output['disp_dim'] = self.disp_dim
                    else:
                        assert False
            # time-wise DAC scores the plan against the drivable area, one score per pose
            if self._config.twdac_check:
                if plan_query_pNC_expanded is None:
                    plan_query_twDAC = motion_query[:, :,-1]   # (L, B, K, D)
                else:
                    plan_query_twDAC = plan_query_pNC_expanded[:,:,0]

                L_, B, K_, C_ = plan_query_twDAC.shape

                if self._config.twdac_bev_module:
                    plan_query_twDAC_ = plan_query_twDAC.flatten(0,1)
                    dac_plan_anchor = [output[f"plan_traj_{lvl + self._config.prop_traj_n_layers}"] for lvl in range(self._config.SWNet_num_layers)]
                    dac_plan_anchor_ = torch.stack(dac_plan_anchor, dim=0)
                    dac_plan_anchor_ = dac_plan_anchor_.flatten(0,1)
                    if dac_plan_anchor_.ndim == 5:
                        dac_plan_anchor_ = dac_plan_anchor_[:,-1]
                    if self._config.twdac_reference_detach:
                        dac_plan_anchor_ = dac_plan_anchor_.detach()
                    dac_bev_feature = self.twdac_BEV_Module(bev_feature)
                    twdac_bev_seg_out = self.twdac_BEV_Seg_Head(dac_bev_feature)
                    dac_bev_feature = dac_bev_feature[None].repeat(L_, 1, 1, 1, 1).flatten(0,1)

                    _, plan_query_twDAC_, _, _ = self.twdac_BEV_decoder(\
                            bev_embed=dac_bev_feature,
                            query=plan_query_twDAC_,
                            output_coord_sigmoid=None,
                            plan_anchor=dac_plan_anchor_,
                            plan_anchor_encoder=None,
                            ego_fut_mode=1,
                            reg_branches=None,
                            plan_reg_branch=None,
                            planning_only=True,
                            status_emb=None,
                        )
                    plan_query_twDAC = plan_query_twDAC_[-1].reshape(L_,B,K_,C_) + plan_query_twDAC

                    output['twdac_bev_seg_out'] = twdac_bev_seg_out
                twdac_pred = self.twdac_heads(plan_query_twDAC).reshape(L, B, K, self._config.num_pose)
                output['twdac_pred'] = twdac_pred

    def _score_at_test_time(self, output, stage):
        """Pick one plan: a weighted sum of the safety heads, or plain imitation."""
        plan_cls = stage['plan_cls']
        plan_reg_pred = stage['plan_reg_pred']

        if not self.training:
            # eval_safedrive.sh sets scoring_test and passes the weights; otherwise take the argmax class
            if self.scoring_test:
                pad_idx = self._config.num_proposal_2stage if self._config.proposalnet_2stage else self._config.num_plan_anchor
                plan_score = self.imi_test_weight * safe_log(plan_cls.softmax(-1))
                NC = self.NC_test_weight * safe_log(output['NC'][-1][:, :pad_idx].sigmoid())
                DAC = self.DAC_test_weight * safe_log(output['DAC'][-1][:, :pad_idx].sigmoid())
                if self.no_EP_TC_sum_scoring:
                    EP = self.EP_test_weight * safe_log(output['EP'][-1][:, :pad_idx].sigmoid())
                    TTC = self.TTC_test_weight * safe_log(output['TTC'][-1][:, :pad_idx].sigmoid())
                    W = self.W_test_weight
                    score = plan_score + NC + DAC + EP + TTC
                else:
                    EP = self.EP_test_weight * output['EP'][-1][:, :pad_idx].sigmoid()
                    TTC = self.TTC_test_weight * output['TTC'][-1][:, :pad_idx].sigmoid()
                    W = self.W_test_weight * safe_log(EP + TTC)
                    score = plan_score + NC + DAC + W
                if self._config.pred_pdm_score:
                    pdm_score = self.pdm_score_test_weight * output['pdm_score'][-1][:, :pad_idx].sigmoid()
                    score += pdm_score
                if self._config.pred_DDC:
                    DDC = self.DDC_test_weight * output['DDC'][-1][:, :pad_idx].sigmoid()
                    score += DDC
                if self._config.pred_TLC:
                    TLC = self.TLC_test_weight * output['TLC'][-1][:, :pad_idx].sigmoid()
                    score += TLC
                if self._config.pred_LK:
                    LK = self.LK_test_weight * output['LK'][-1][:, :pad_idx].sigmoid()
                    score += LK

                # a plan is only compliant if every pose is: multiply over time
                if self.twdac_scoring:
                    twdac = output['twdac_pred'][-1].sigmoid()       # (B, K, T)
                    twdac = twdac.prod(dim=-1)                       # (B, K): every pose must comply

                    if self.TwDAC_bevseg_pred:
                        # weight the head by what the predicted BEV segmentation says
                        plan_traj = output[f"plan_traj_{self.prop_traj_n_layers + self._config.SWNet_num_layers - 1}"]
                        bev_seg_prob = get_seg_prob(
                            bev_segmap=output['twdac_bev_seg_out'].sigmoid(),
                            plan_traj=plan_traj,
                            reduction='min',
                            interpolation=True,
                            class_idx=1,
                            num_points=9,
                            bbox_margin=self.TwDAC_bbox_margin
                        )                                            # (B, K, T)
                        twdac = bev_seg_prob.min(dim=-1)[0] * twdac  # worst pose, (B, K)

                    time_wise_DAC_pred_score = self.TwDAC_test_weight * safe_log(twdac)
                    score += time_wise_DAC_pred_score

                # and collision-free only if it is against every agent at every step
                if self.pair_NC_scoring:
                    pair_NC = output['pwnc_pred'][-1].sigmoid()
                    pair_NC[~output['agent_valid_mask']] = 1   # padded slots must not lower the product
                    pair_NC = pair_NC.prod(dim=-1)
                    min_pair_NC_vals = pair_NC.prod(dim=1)

                    score += self.PwNC_test_weight * safe_log(min_pair_NC_vals)

                mode_idx = score.argmax(-1)
            else:
                mode_idx = plan_cls.argmax(dim=-1)

            mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._config.trajectory_sampling.num_poses,3)
            best_reg = torch.gather(plan_reg_pred, 1, mode_idx).squeeze(1)   # (B, T, 3)

            output['trajectory'] = best_reg
            if self.scoring_test:
                output['selected_idx'] = score.argmax(-1)

    def select_topk(self,
                    agent_boxes: torch.Tensor,
                    plan_traj: torch.Tensor,
                    valid_agent_mask: torch.Tensor,
                    agent_sigmoid: bool = False):
        """Per plan anchor, the nearest valid agents."""
        B, max_agents, _ = agent_boxes.shape
        N_anchor = plan_traj.shape[1]

        K = self._config.num_filtering_instance
        # torch.topk requires k <= A', hence k_eff
        k_eff = min(K, max_agents)

        if max_agents == 0:
            topk_idx = agent_boxes.new_zeros((B, 0, N_anchor), dtype=torch.long)
            topk_valid = agent_boxes.new_zeros((B, 0, N_anchor), dtype=torch.bool)
            return topk_idx, topk_valid

        # agent_boxes: [B, A', C], plan_anchor: [B, P, T, 3]
        agents_xy = agent_boxes[..., None, None, :2]      # (B, A', 1, 1, 2)
        pa_xy = plan_traj[:, None, ..., :2]           # (B, 1, P, T, 2)
        if self._config.agent_filtering_fix and agent_sigmoid:
            x0, x1, _ = self._config.grid_config['x']; y0, y1, _ = self._config.grid_config['y']
            agents_xy = agents_xy.clone()
            agents_xy[..., 0:1] = agents_xy[..., 0:1] * (x1 - x0) + x0
            agents_xy[..., 1:2] = agents_xy[..., 1:2] * (y1 - y0) + y0

        diff = agents_xy - pa_xy                        # (B, A', P, T, 2)
        l2 = torch.norm(diff, dim=-1).min(-1)[0]        # [B, A', P]  (min over T)

        # only rows that survived the det-score filter take part in the top-k;
        # padded rows are masked with +inf so they can never be selected
        valid_A = valid_agent_mask.unsqueeze(-1).expand(-1, -1, N_anchor)  # (B, A', P)
        l2 = l2.masked_fill(~valid_A, float('inf'))

        topk_idx = l2.topk(k=k_eff, dim=1, largest=False).indices        # (B, k_eff, P)
        topk_valid = torch.gather(valid_A, 1, topk_idx)                  # (B, k_eff, P)

        if k_eff < K:
            pad_k = K - k_eff
            pad_idx = topk_idx.new_zeros(B, pad_k, N_anchor)             # (B, pad_k, P)
            pad_valid = topk_valid.new_zeros(B, pad_k, N_anchor)         # (B, pad_k, P)
            topk_idx = torch.cat([topk_idx, pad_idx], dim=1)             # (B, K, P)
            topk_valid = torch.cat([topk_valid, pad_valid], dim=1)       # (B, K, P)
        return topk_idx, topk_valid

    def pack_selected(self, instance_query, output_coord_sigmoid, mask, agent_trajs=None):
        # instance_query:        [B, N_agent, F]
        # output_coord_sigmoid:  [B, N_agent, C]
        # agent_trajs:          [B, N_agent, 9, 3]
        """Compact the surviving instances to the front, padding the rest."""
        B, N_agent, F = instance_query.shape
        _, _, C = output_coord_sigmoid.shape
        T = agent_trajs.shape[2] if agent_trajs is not None else 0

        num_true = mask.sum(dim=1)                 # [B]
        max_true = int(num_true.max().item())

        if max_true == 0:
            padded_queries = instance_query.new_zeros((B, 0, F))
            padded_coords  = instance_query.new_zeros((B, 0, C))
            padded_trajs = agent_trajs.new_zeros((B, 0, T, 3)) if agent_trajs is not None else None
            sorted_idx = instance_query.new_full((B, 0), fill_value=N_agent, dtype=torch.long)
            valid_mask = sorted_idx != N_agent
            return padded_queries, padded_coords, padded_trajs, sorted_idx, valid_mask

        idx = torch.arange(N_agent, device=mask.device)[None, :].expand(B, -1)   # (B, N_agent)
        if mask.ndim == 3:
            mask = mask.sum(-1).bool()
        masked_idx = torch.where(mask, idx, torch.full_like(idx, N_agent))       # (B, N_agent)

        sorted_idx = masked_idx.sort(dim=1, descending=False).values[:, :max_true]   # (B, max_true)
        valid_mask = (sorted_idx != N_agent)                                         # [B, max_true] (bool)

        # dummy rows map to 0 to keep the gather in bounds
        safe_idx_q = torch.where(valid_mask, sorted_idx, torch.zeros_like(sorted_idx))  # (B, max_true)
        safe_idx_q = safe_idx_q.unsqueeze(-1).expand(-1, -1, F)                         # (B, max_true, F)
        safe_idx_c = safe_idx_q[:, :, :C]                                               # (B, max_true, C)

        # gather
        gathered_q = instance_query.gather(1, safe_idx_q)          # (B, max_true, F)
        gathered_c = output_coord_sigmoid.gather(1, safe_idx_c)    # (B, max_true, C)

        padded_queries = gathered_q.masked_fill(~valid_mask.unsqueeze(-1), 0)
        padded_coords  = gathered_c.masked_fill(~valid_mask.unsqueeze(-1), 0)

        if agent_trajs is not None:
            safe_idx_m = safe_idx_q[:, :, :T]                                               # (B, max_true, T)
            gathered_m = agent_trajs.gather(1, safe_idx_m[...,None].expand(-1, -1, -1, 3))            # (B, max_true, T, 3)
            padded_trajs = gathered_m.masked_fill(~valid_mask.unsqueeze(-1).unsqueeze(-1), 0)
        else:
            padded_trajs = None

        return padded_queries, padded_coords, padded_trajs, sorted_idx, valid_mask
