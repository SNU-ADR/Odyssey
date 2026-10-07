from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class SafeDrive_Config:
    """Global TransFuser config."""

    grid_config: Dict[str, Any] = field(default_factory=lambda: {
        'x': [0, 32, 1.0],
        'y': [-32, 32, 1.0],
        'z': [-5, 3, 8],
        'depth' : [1.0, 45.0, 0.5],
        'downsample': 16
    })
    neck_in_channels: list = field(default_factory=lambda: [64, 128, 256, 512])
    neck_upsample_strides: list = field(default_factory=lambda: [0.25, 0.5, 1, 2])
    neck_out_channels_second_fpn_channels: list = field(default_factory=lambda: [64, 64, 64, 64])

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    image_architecture: str = "resnet34"
    #: Local ImageNet weights for image_architecture, used only if timm cannot fetch them from the
    #: Hugging Face hub (see SafeDrive_Backbone). Empty = no fallback: that case raises instead.
    bkb_path: str = ""

    latent_rad_thresh: float = 4 * np.pi / 9

    # False => camera-only: no lidar feature is cached or loaded, the SECOND voxel branch is not
    # built, and the BEV encoder must not contain 'lidar_cross_attn' in its operation_order
    # (see the *_CamOnly agent configs). The lidar_* ranges below still define the BEV grid.
    use_lidar: bool = True

    # True => the detection head classifies vehicles and pedestrians (2 sigmoid logits per query);
    # pedestrians are then matched and supervised, and the denoising queries and SWNet instance
    # selection include them. The paper model detects vehicles only (1 logit): pedestrian boxes are
    # in every cached target (agent_labels 1) but are dropped before Hungarian matching. baseline and
    # sdroute set it (see the *_CamOnly_Ped and safedrive_baseline_agent configs).
    det_pedestrian: bool = False

    # True  => the feature cache holds the preprocessed uint8 images (3 frames x 3 cams x 3x256x512,
    #          ~2.8 MB/sample gzipped on top of the targets, ~360 GB for navtrain).
    # False => the cache holds only the image paths + calibration (CameraPaths); the dataloader
    #          worker decodes/undistorts/resizes at collate time (~0.35 s CPU per sample), so the
    #          cache shrinks to the targets (~0.7 MB/sample). Same pixels either way. Only read
    #          when features are computed (dataset caching, or online evaluation); training reads
    #          whatever the cache contains.
    cache_camera_images: bool = True

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    # new
    lidar_seq_len: int = 1

    lidar_resolution_width: int = 256
    lidar_resolution_height: int = 256

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 12.0
    trajectory_cls_weight: float = 10.0
    trajectory_reg_weight: float = 8.0
    agent_class_weight: float = 5.0
    agent_box_weight: float = 2.0
    agent_vel_weight: float = 1.0
    bev_semantic_weight: float = 14.0
    fut_bev_semantic_weight: float = 14.0
    prediction_loss_weight: float = 1.0

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: ("linestring", [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]),  # centerline
        4: (
            "box",
            [
                TrackedObjectType.CZONE_SIGN,
                TrackedObjectType.BARRIER,
                TrackedObjectType.TRAFFIC_CONE,
                TrackedObjectType.GENERIC_OBJECT,
            ],
        ),  # static_objects
        5: ("box", [TrackedObjectType.VEHICLE]),  # vehicles
        6: ("box", [TrackedObjectType.PEDESTRIAN]),  # pedestrians
    }

    bev_pixel_width: int = lidar_resolution_width
    bev_pixel_height: int = lidar_resolution_height // 2
    bev_pixel_size: float = 0.25

    num_bev_classes = 7
    bev_features_channels: int = 256

    epoch: int = 100
    _warm_epoch: int = 1

    #### BEVFormer #####
    bevformer_encoder: Dict[str, Any] = field(default_factory=lambda: {
        "type": "BEVFormerEncoder",
        "num_layers": 3,
        "pc_range": [0, -32.0, -3.0, 32.0, 32.0, 5.0],
        "num_points_in_pillar": 4,
        "return_intermediate": False,
        "WH_reverse": True,
        "transformerlayers": {
            "type": "BEVFormerLayer",
            "attn_cfgs": [
                {
                    "type": "TemporalSelfAttention",
                    "embed_dims": 256,
                    "num_levels": 1,
                },
                {
                    "type": "MSDeformableAttention3D",
                    "embed_dims": 256,
                    "num_points": 8,
                    "num_levels": 1,
                },
                {
                    "type": "SpatialCrossAttention",
                    "pc_range": [0, -32.0, -3.0, 32.0, 32.0, 5.0],
                    "embed_dims": 256,
                    "deformable_attention": {
                        "type": "MSDeformableAttention3D",
                        "embed_dims": 256,
                        "num_points": 8,
                        "num_levels": 1,
                    },
                },
            ],
            "feedforward_channels": 512,
            "ffn_dropout": 0.1,
            "operation_order": ['self_attn', 'norm', 'lidar_cross_attn', 'cross_attn', 'norm', 'ffn', 'norm'],
        },
    })

    ins_decoder_layer: int = 4
    bevformer_decoder: Dict[str, Any] = field(default_factory=lambda: {
        "type": "DetectionTransformerDecoder",
        "num_layers": 4,
        "return_intermediate": True,
        "transformerlayers": {
            "type": "DetrTransformerDecoderLayer",
            "attn_cfgs": [
                {
                    "type": "MultiheadAttention",
                    "embed_dims": 256,
                    "num_heads": 8,
                    "dropout": 0.1,
                },
                {
                    "type": "CustomMSDeformableAttention",
                    "embed_dims": 256,
                    "num_levels": 1,
                },
            ],
            "feedforward_channels": 1024,
            "ffn_dropout": 0.1,
            "operation_order": [
                "self_attn", "norm", "cross_attn", "norm",
                "ffn", "norm"
            ],
        },
    })

    positional_encoding: Dict[str, Any] = field(default_factory=lambda: {
        "type": "LearnedPositionalEncoding",
        "num_feats": 128,
        "row_num_embed": 64,
        "col_num_embed": 64,
    })

    num_reg_fcs: int = 2

    bev_backbone_input_dim: int = 256
    bev_backbone_output_dim: list = field(default_factory=lambda: [256, 512, 1024])
    bev_h: int = 64
    bev_w: int = 32
    bev_neck_in_channels: list = field(default_factory=lambda: [256, 512, 1024])
    bev_neck_upsample_strides: list = field(default_factory=lambda: [2, 4, 8])
    bev_out_channels_second_fpn_channels: list = field(default_factory=lambda: [128,128,128])
    num_input_frames: int = 3
    x_range: float = 32.
    y_range: float = 64.

    SWNet_num_layers: int = 6
    SWNet_cfg: Dict[str, Any] = field(default_factory=lambda: {
        "type": "Motion_Decoder",
        "num_layers": 6,
        "return_intermediate": True,
        "transformerlayers": {
            "type": "Motion_Decoder_Layer",
            "attn_cfgs": [
                {
                    "type": "MultiheadAttention",
                    "embed_dims": 256,
                    "num_heads": 8,
                    "dropout": 0.1,
                },
                {
                    "type": "Traj_Guided_Deform_Attention",
                    "embed_dims": 256,
                    "num_levels": 1,
                    "num_frame": 9,
                },
            ],
            "feedforward_channels": 1024,
            "ffn_dropout": 0.1,
            "operation_order": [
                "self_attn", "norm", "cross_attn", "norm",
                "ffn", "norm"
            ],
        },
    })

    num_plan_anchor: int = 8192
    plan_anchor_path: str = 'trajectory_anchors/trajectory_anchors_256_GTRS.npy'

    pc_range: list = field(default_factory=lambda: [0, -32, -5, 32, 32, 3])
    voxel_size: list = field(default_factory=lambda: [0.125, 0.125, 0.2])

    num_pose: int = 8

    safety_scoring_ray: bool = False
    scene_level_safety: bool = False

    NC_loss_weight: float = 3.0
    DAC_loss_weight: float = 3.0
    TTC_loss_weight: float = 4.0
    EP_loss_weight: float = 2.0
    C_loss_weight: float = 1.0
    pdm_score_loss_weight: float = 1.0
    DDC_loss_weight: float = 1.0
    TLC_loss_weight: float = 1.0
    LK_loss_weight: float = 1.0

    safety_score_weight: float = 1.0
    pred_pdm_score: bool = False
    pred_DDC: bool = False
    pred_TLC: bool = False
    pred_LK: bool = False
    EP_loss_no_NC_DAC: bool = False

    imi_test_weight: float = 1.0
    NC_test_weight: float = 0.0
    DAC_test_weight: float = 0.0
    EP_test_weight: float = 0.0
    TTC_test_weight: float = 0.0
    C_test_weight: float = 0.0
    pdm_score_test_weight: float = 0.0
    W_test_weight: float = 0.0
    DDC_test_weight: float = 0.0
    TLC_test_weight: float = 0.0
    LK_test_weight: float = 0.0
    HC_test_weight: float = 0.0
    TwDAC_test_weight: float = 0.0
    PwNC_test_weight: float = 0.0

    # === Rollout safety GT: score the model's own plans with the PDM simulator ===
    # Master switch. Phase 2 and 3 turn this on; the whole block below is inert without it.
    use_target_scores: bool = False
    # "pdms" -> 7-col GT [NC,DAC,EP,TTC,comfort,DDC,final];
    # "epdms" -> 9-col GT [NC,DAC,EP,TTC,comfort,DDC,TLC,LK,final] (adds TLC/LK heads).
    safety_score_mode: str = "pdms"
    # dir holding metadata/*.csv of per-token metric_cache.pkl (world cache, NOT scores).
    # Loaded by navsim.common.dataloader.MetricCacheLoader (reads the CSV, token=path[-2]).
    safety_metric_cache_path: str = ""
    # Ray CPU workers for the live PDM rollout (0 => run inline without Ray).
    safety_ray_threads: int = 8

    AF_topk: bool = False
    num_filtering_instance: int = 20
    AF_det_score: bool = False
    ins_filter_cls_thr: float = 0.3

    no_planning: bool = False
    bev_seg_multi_class: bool = False
    future_bev_frames: int = 4

    num_vehicle_bounding_boxes: int = 50
    num_pedestrian_bounding_boxes: int = 100

    # dn
    dn_detection: bool = False
    dn_scalar: int = 5
    dn_bbox_noise_scale: float = 0.05
    box_size_nose: float = 0.0

    pwnc_check: bool = False
    pair_NC_loss_weight: float = 1.0
    pwnc_focal_loss: bool = False
    pair_NC_class_weighting: bool = False
    pwnc_focal_alpha: float = 0.25

    pwdisp_check: bool = False
    pair_Disp_loss_weight: float = 1.0
    pair_Disp_motion_GT_loss: bool = False
    pair_Disp_motion_with_yaw: bool = False

    twdac_check: bool = False
    twdac_loss_weight: float = 1.0
    twdac_bev_module: bool = False
    twdac_reference_detach: bool = False
    time_wise_DAC_BEV_module_layer: int = 1.0
    twdac_bev_deform_cfg: Dict[str, Any] | None = None
    twdac_focal_loss: bool = False
    time_size_DAC_focal_alpha: float = 0.75

    agent_name: str = "temp"
    low_lr_scale: float = 0.1
    # Phase 2: freeze perception and train only the planning / safety heads.
    # freeze_perception_prefixes: module-name prefixes to freeze (None -> built-in list).
    freeze_perception: bool = False
    freeze_perception_prefixes: Optional[List[str]] = None
    finetune_markers: Optional[List[str]] = None
    det_coord_detach: bool = False
    agent_filtering_fix: bool = False
    all_motion_predidction_loss: bool = False

    # === SD-route conditioning ===
    # Off by default: no route module is built, the target builder emits no route, and the
    # decoders keep the base config's state_dict keys. Turning it on also requires use_sdroute
    # and 'route_attn' in each plan decoder's transformerlayers (see safedrive_sdroute_agent.yaml);
    # this flag alone only builds the shared encoder and caches the route.
    use_sdroute: bool = False
    sdroute_num_heads: int = 8
    sdroute_dropout: float = 0.0
    # The route replaces the 4-way driving_command, which is the first 4 channels of
    # status_feature here ([command(4), vel(2), accel(2)]), so it is sliced off with [..., 4:].
    drop_driving_command: bool = False

    # ProposalNet trajectory decoder
    proposal_traj_cfg: Dict[str, Any] | None = None
    prop_traj_n_layers: int = 0
    stage1_reference_points_detach: bool = False

    proposalnet_2stage: bool = False
    num_proposal_2stage: int = 256
    imi_2stage_weight: float = 0.5
    NC_2stage_weight: float = 25.0
    DAC_2stage_weight: float = 15.0
    EP_2stage_weight: float = 10.0
    TTC_2stage_weight: float = 10.0
    W_2stage_weight: float = 20.0
    pdm_score_2stage_weight: float = 0.0
    DDC_2stage_weight: float = 0.0
    TLC_2stage_weight: float = 0.0
    LK_2stage_weight: float = 0.0


    # denoising queries
    dn_detection: bool = False
    dn_scalar: int = 5

    margin_type: str = 'fixed'
    ins_box_margin: Tuple[float,float] = (1.0, 1.0)
    ego_box_margin: Tuple[float,float] = (1.0, 1.0)

    # Diffusion decoder

    # dense world model

    # 1stage_pwnc

    # Set TwDAC BEV resolution

    # Transfuser
    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    gpt_linear_layer_init_mean = 0.0
    gpt_linear_layer_init_std = 0.02
    gpt_layer_norm_init_weight = 1.0
    perspective_downsample_factor = 1
    transformer_decoder_join = True

    vis_save_name: str = ""

    # Recompute lidar2img on the fly: fixes the non-uniform x/y resize and the
    # post translation that belonged in the Z column.

    # optmizer
    lr_down: str = "image_encoder"   # modules trained at the reduced lr
    weight_decay: float = 1e-4
    lr_steps = [70]
    optimizer_type = "AdamW"
    scheduler_type: str = "cos"
    cfg_lr_mult = 0.5
    opt_paramwise_cfg = {
        "name":{
            "image_encoder":{
                "lr_mult": cfg_lr_mult
            }
        }
    }

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
