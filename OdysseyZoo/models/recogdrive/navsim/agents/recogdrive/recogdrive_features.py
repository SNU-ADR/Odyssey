from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import numpy as np
import gzip
import pickle
from PIL import Image

from navsim.agents.abstract_agent import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import Scene, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from .recogdrive_backbone import RecogDriveBackbone
from .route_text import route_to_text
from .utils.internvl_preprocess import load_image

def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


NAVIGATION_COMMANDS = ['turn left', 'go straight', 'turn right']


def command_text(high_command_one_hot: torch.Tensor) -> str:
    """The dataset's driving command as prompt text. NAVSIM's one-hot is [left, straight, right,
    unknown]; the last slot, or no hot slot at all, reads "unknown". Every prompt that carries the
    command goes through here, so the cache, online evaluation and the route-text fallback agree."""
    return next((NAVIGATION_COMMANDS[i] for i, v in enumerate(high_command_one_hot[:3]) if v == 1), "unknown")


# What fills "3. Active navigation command: [...]": the dataset's 3-way command (released model),
# or SimLingo-style text from the SD route, e.g. [TURN RIGHT IN 50 M, THEN TURN LEFT].
NAV_PROMPTS = ("command", "sdroute_text")
# Left-padding length of the VLM input (backbone max_length). Route text can push a prompt past the
# released 2800, and an unpadded longer sequence would break the fixed length, so sdroute_text pads
# to 2832. Real tokens keep their hidden states: position ids come from the attention mask.
PROMPT_MAX_LENGTH = {"command": 2800, "sdroute_text": 2832}
OUTPUT_REQUIREMENTS = (
    "\nOutput requirements:\n- Predict 8 future trajectory points\n"
    "- Each point format: (x:float, y:float, heading:float)\n"
    "- Use [PT, ...] to encapsulate the trajectory\n"
    "- Maintain numerical precision to 2 decimal places"
)


def build_question(history_trajectory: torch.Tensor, navigation: str) -> str:
    """The VLM question. The hidden-state cache and online evaluation both build it here, so the two
    can never drift apart. With navigation = a 3-way command this is byte-identical to upstream
    ReCogDrive's planner prompt (hidden-state cache and online path)."""
    history_str = " ".join([
        f'   - t-{3-i}: ({format_number(history_trajectory[i, 0].item())}, '
        f'{format_number(history_trajectory[i, 1].item())}, '
        f'{format_number(history_trajectory[i, 2].item())})'
        for i in range(history_trajectory.shape[0])
    ])
    prompt = (
        "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
        "1. Visual perception from front camera view\n"
        f"2. Historical motion context (last 4 timesteps):{history_str}"
        f"\n3. Active navigation command: [{navigation.upper()}]"
    )
    return f"{prompt}{OUTPUT_REQUIREMENTS}"


def load_cached_route(route_cache_path: str, log_name: str, token: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """route_centerline (P, 5) and route_centerline_mask (P,) from <root>/<log>/<token>/sdroute_target.gz."""
    path = Path(route_cache_path) / log_name / token / "sdroute_target.gz"
    if not path.exists():
        raise FileNotFoundError(f"no cached route for token {token}: {path}")
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    return data["route_centerline"], data["route_centerline_mask"]


def sdroute_navigation(route: torch.Tensor, mask: torch.Tensor, high_command_one_hot: torch.Tensor) -> str:
    """Route text for the navigation slot. A route too short to describe keeps the dataset command,
    the only navigation left for that sample."""
    text = route_to_text(route, mask)
    if text is not None:
        return text
    return command_text(high_command_one_hot)


class ReCogDriveFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self,
                 cache_hidden_state: bool = True,
                 model_type: Optional[str] = None,
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda",
                 cache_mode: bool = False,
                 nav_prompt: str = "command",
                 route_cache_path: Optional[str] = None,
                 hidden_state_dtype: str = "float32", ):
        """
        Initializes the feature builder.

        Args:
            cache_hidden_state (bool): If True, operates in online mode, initializes the backbone,
                                       and computes the hidden state. If False, operates in offline
                                       mode, does not initialize the backbone, and returns
                                       pre-computable tensors, including a tensorized representation
                                       of the image file path.
            model_type (str, optional): The type of model to load ('internvl' or 'qwen'). Required if cache_hidden_state is True.
            checkpoint_path (str, optional): Path to the model checkpoint. Required if cache_hidden_state is True.
            device (str): The device to load the model onto.
            nav_prompt (str): What fills the navigation slot of the prompt, one of NAV_PROMPTS.
            route_cache_path (str, optional): Root of <log>/<token>/sdroute_target.gz. Required for
                                              nav_prompt='sdroute_text'.
            hidden_state_dtype (str): dtype of the cached last_hidden_state, 'float32' or 'bfloat16'.
                                      'bfloat16' is lossless (the VLM runs in bf16) and halves loader RAM.
        """
        super().__init__()
        if nav_prompt not in NAV_PROMPTS:
            raise ValueError(f"nav_prompt must be one of {NAV_PROMPTS}, got {nav_prompt!r}")
        if hidden_state_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"hidden_state_dtype must be 'float32' or 'bfloat16', got {hidden_state_dtype!r}")
        self.hidden_state_dtype = getattr(torch, hidden_state_dtype)
        self.cache_hidden_state = cache_hidden_state
        self.backbone = None
        self.cache_mode = cache_mode
        self.nav_prompt = nav_prompt
        self.route_cache_path = route_cache_path
        # Dataset._cache_scene_with_token passes the Scene to builders that set this: the route is
        # looked up by token, which AgentInput does not carry.
        self.requires_scene = nav_prompt == "sdroute_text"

        if self.cache_hidden_state and self.cache_mode:
            if not model_type or not checkpoint_path:
                raise ValueError("In online mode (cache_hidden_state=True), `model_type` and `checkpoint_path` must be provided.")
            self.backbone = RecogDriveBackbone(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                device=device
            )

    def get_unique_name(self) -> str:
        return "internvl_feature"

    def navigation(self, high_command_one_hot: torch.Tensor, scene: Optional[Scene] = None) -> str:
        """Text for the navigation slot of build_question."""
        if self.nav_prompt == "command":
            return command_text(high_command_one_hot)
        if scene is None:
            raise RuntimeError(
                "nav_prompt='sdroute_text' needs the Scene to look up the route by token; "
                "Dataset._cache_scene_with_token passes it to builders with requires_scene"
            )
        if not self.route_cache_path:
            raise RuntimeError("nav_prompt='sdroute_text' needs agent.sdroute_cache_path (the route cache root)")
        meta = scene.scene_metadata
        route, mask = load_cached_route(self.route_cache_path, meta.log_name, meta.initial_token)
        return sdroute_navigation(route, mask, high_command_one_hot)

    def compute_features(self, agent_input: AgentInput, scene: Optional[Scene] = None) -> Dict[str, torch.Tensor]:

        ego_statuses = agent_input.ego_statuses
        cameras = agent_input.cameras

        history_trajectory = torch.tensor(
            [[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego_statuses[:4]],
            dtype=torch.float32
        )
        high_command_one_hot = torch.tensor(ego_statuses[-1].driving_command, dtype=torch.float32)
        status_feature = torch.cat([
            high_command_one_hot.clone(),
            torch.tensor(ego_statuses[-1].ego_velocity, dtype=torch.float32),
            torch.tensor(ego_statuses[-1].ego_acceleration, dtype=torch.float32)
        ], dim=-1)


        if not self.cache_hidden_state:
            image_path = str(cameras[-1].cam_f0.image)
            
            path_as_ordinals = [ord(char) for char in image_path]
            
            path_tensor = torch.tensor(path_as_ordinals, dtype=torch.long)
            
            return {
                "history_trajectory": history_trajectory.cpu(),
                "high_command_one_hot": high_command_one_hot.cpu(),
                "status_feature": status_feature.cpu(),
                "image_path_tensor": path_tensor.cpu(),
            }
        else:
            if self.backbone is None:
                raise RuntimeError("FeatureBuilder is in online mode, but the backbone was not initialized.")
            
            pixel_values = load_image(str(cameras[-1].cam_f0.image),max_num=12).unsqueeze(0)

            pixel_values_squeezed = pixel_values.squeeze(1)
            num_patches_list = [pv.shape[0] for pv in pixel_values_squeezed]
            pixel_values_cat = torch.cat(list(pixel_values_squeezed), dim=0)

            questions = [build_question(history_trajectory, self.navigation(high_command_one_hot, scene))]

            # Only the values go to disk. Without no_grad every sample also builds the autograd graph
            # (the VLM params require grad), and a caller that keeps the result keeps that graph on GPU.
            with torch.no_grad():
                outputs = self.backbone(pixel_values_cat.cuda(), questions, num_patches_list=num_patches_list,
                                        max_length=PROMPT_MAX_LENGTH[self.nav_prompt])
            last_hidden_state = outputs.hidden_states[-1]

            return {
                "history_trajectory": history_trajectory.cpu(),
                "high_command_one_hot": high_command_one_hot.cpu(),
                "last_hidden_state": last_hidden_state.squeeze(0).to(self.hidden_state_dtype).cpu(),
                "status_feature": status_feature.cpu(),
            }


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}


class SDRouteTargetBuilder(AbstractTargetBuilder):
    """Cache slot for the SD route: <token>/sdroute_target.gz holding
    route_centerline (120, 5) float32 and route_centerline_mask (120,) bool.

    Cache-only. The route is built once by scripts/sdroute/run_sdroute_cache.sh with the shared
    OdysseyZoo/sdroute builder, so training reads exactly the tensor the other SD-route models use.
    """

    def get_unique_name(self) -> str:
        return "sdroute_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        raise RuntimeError(
            "sdroute_target is cache-only: generate sdroute_target.gz with scripts/sdroute/run_sdroute_cache.sh "
            "and train with use_cache_without_dataset=True"
        )
