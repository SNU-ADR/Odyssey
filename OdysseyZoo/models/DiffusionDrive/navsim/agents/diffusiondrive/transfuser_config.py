from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

# Repo root = three levels above navsim/agents/diffusiondrive/. Asset defaults resolve against it,
# not the cwd, so every entry point finds the vendored files from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TransfuserConfig:
    """Global TransFuser config."""

    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"
    #: Local ImageNet weights for image_architecture, used only if timm cannot fetch them from the
    #: Hugging Face hub (see TransfuserBackbone). Empty = no fallback: that case raises instead.
    bkb_path: str = ""
    #: 20 k-means trajectory anchors (20, 8, 2). The vendored file is the one both shipped ckpts
    #: were trained with (it is also stored in each ckpt as _trajectory_head.plan_anchor).
    plan_anchor_path: str = str(_REPO_ROOT / "assets" / "kmeans_navsim_traj_20.npy")

    latent: bool = False
    latent_rad_thresh: float = 4 * np.pi / 9

    max_height_lidar: float = 100.0
    pixels_per_meter: float = 4.0
    hist_max_per_pixel: int = 5

    lidar_min_x: float = -32
    lidar_max_x: float = 32
    lidar_min_y: float = -32
    lidar_max_y: float = 32

    lidar_split_height: float = 0.2
    use_ground_plane: bool = False

    # new
    lidar_seq_len: int = 1

    camera_width: int = 1024
    camera_height: int = 256
    lidar_resolution_width = 256
    lidar_resolution_height = 256

    img_vert_anchors: int = 256 // 32
    img_horz_anchors: int = 1024 // 32
    lidar_vert_anchors: int = 256 // 32
    lidar_horz_anchors: int = 256 // 32

    block_exp = 4
    n_layer = 2  # Number of transformer layers used in the vision backbone
    n_head = 4
    n_scale = 4
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    # Mean of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_mean = 0.0
    # Std of the normal distribution initialization for linear layers in the GPT
    gpt_linear_layer_init_std = 0.02
    # Initial weight of the layer norms in the gpt.
    gpt_layer_norm_init_weight = 1.0

    perspective_downsample_factor = 1
    transformer_decoder_join = True
    detect_boxes = True
    use_bev_semantic = True
    use_semantic = False
    use_depth = False
    add_features = True

    # Transformer
    tf_d_model: int = 256
    tf_d_ffn: int = 1024
    tf_num_layers: int = 3
    tf_num_head: int = 8
    tf_dropout: float = 0.0

    # ----- SD-route arm (see agents/sdroute/route_attention.py) -----
    # Folds the cached (120,5) ego-local route polyline into 24 segment tokens (shared encoder) and
    # adds a zero-gated cross-attention inside each diffusion decoder layer
    # (CustomTransformerDecoderLayer), placed after cross_ego and before the FFN: the trajectory
    # query reads the route tokens there. Zero-init gate -> exact no-op at step 0.
    use_sdroute: bool = False
    sdroute_num_heads: int = 8
    sdroute_dropout: float = 0.0
    #: Fixed seed for the route module's own init, so use_sdroute by itself leaves every
    #: other module's init bit-identical at a given global seed (drop_driving_command does not).
    sdroute_init_seed: int = 0
    #: Drop the driving_command(4) slice of status_feature = [command(4), velocity(2),
    #: accel(2)] at ingest and narrow _status_encoding from Linear(8,D) to Linear(4,D), so
    #: the SD route (use_sdroute) replaces the 4-way command rather than riding alongside
    #: it. Changes parameter shapes, so it is not checkpoint-compatible with a command-on run.
    drop_driving_command: bool = False

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 12.0
    trajectory_cls_weight: float = 10.0
    trajectory_reg_weight: float = 8.0
    diff_loss_weight: float = 20.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 14.0
    use_ema: bool = False
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
    bev_features_channels: int = 64
    bev_down_sample_factor: int = 4
    bev_upsample_factor: int = 2


    # optmizer
    weight_decay: float = 1e-4
    lr_steps = [70]
    optimizer_type = "AdamW"
    scheduler_type = "MultiStepLR"
    cfg_lr_mult = 0.5
    opt_paramwise_cfg = {
        "name":{
            "image_encoder":{
                "lr_mult": cfg_lr_mult
            }
        }
    }
    # optimizer=dict(
    #     type="AdamW",
    #     lr=1e-4,
    #     weight_decay=1e-6,
    # )
    # scheduler=dict(
    #     type="MultiStepLR",
    #     milestones=[90],
    #     gamma=0.1,
    # )

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
