from dataclasses import dataclass
from typing import Tuple

import numpy as np
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer


@dataclass
class TransfuserConfig:
    """Global TransFuser config."""

    image_architecture: str = "resnet34"
    lidar_architecture: str = "resnet34"

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

    # ----- SD-route conditioning (default-off; see agents/sdroute/) -----
    # The cached (120,5) ego-local route polyline is folded into 24 segment tokens, read by one
    # gated cross-attention inside every decoder layer (zero-init gate, so an untrained module
    # is an exact no-op).
    use_sdroute: bool = False
    sdroute_num_heads: int = 8
    # Fixed seed for the route module's own init, so turning the route on leaves every
    # other module's weights bit-identical at a given global seed.
    sdroute_init_seed: int = 0
    # Condition only the trajectory query; the 30 detection queries do not see the route.
    # True conditions all queries.
    sdroute_on_all_queries: bool = False
    # Drop the driving_command(4) slice of status_feature = [command(4), velocity(2),
    # accel(2)], making the route the only routing signal reaching the planner. This
    # narrows _status_encoding from Linear(8,D) to Linear(4,D), so checkpoints are not
    # compatible with a command-on run, and it shifts the init of later weights.
    drop_driving_command: bool = False

    # detection
    num_bounding_boxes: int = 30

    # loss weights
    trajectory_weight: float = 10.0
    agent_class_weight: float = 10.0
    agent_box_weight: float = 1.0
    bev_semantic_weight: float = 10.0

    # BEV mapping
    bev_semantic_classes = {
        1: ("polygon", [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION]),  # road
        2: ("polygon", [SemanticMapLayer.WALKWAYS]),  # walkways
        3: (
            "linestring",
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
        ),  # centerline
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

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_radius(self) -> float:
        values = [
            self.lidar_min_x,
            self.lidar_max_x,
            self.lidar_min_y,
            self.lidar_max_y,
        ]
        return max([abs(value) for value in values])
