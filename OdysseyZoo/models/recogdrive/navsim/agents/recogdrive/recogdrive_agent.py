from typing import Any, List, Dict, Optional, Union
import os
import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig, OmegaConf
from transformers.feature_extraction_utils import BatchFeature
import math

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import build_from_configs
from .recogdrive_features import (
    ReCogDriveFeatureBuilder, TrajectoryTargetBuilder, SDRouteTargetBuilder,
    NAV_PROMPTS, PROMPT_MAX_LENGTH, build_question, command_text, load_cached_route, sdroute_navigation,
)
from .recogdrive_backbone import RecogDriveBackbone
from .recogdrive_diffusion_planner import (
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)


class ReCogDriveAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        vlm_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        cam_type: Optional[str] = 'single', 
        vlm_type: Optional[str] = 'internvl', 
        dit_type: Optional[str] = 'small', 
        sampling_method: Optional[str] = 'ddim', 
        cache_mode: bool = False, 
        cache_hidden_state: bool = True, 
        lr: float = 1e-4,
        grpo: bool = False,
        metric_cache_path: Optional[str] = '', 
        reference_policy_checkpoint: Optional[str] = '', 
        vlm_size: Optional[str] = 'small',
        train_backbone: bool = False,
        use_sdroute: bool = False,
        sdroute_cache_path: Optional[str] = None,
        sdroute_init_seed: int = 0,
        drop_command: bool = False,
        nav_prompt: str = "command",
        hidden_state_dtype: str = "float32",
    ):
        if nav_prompt not in NAV_PROMPTS:
            raise ValueError(f"agent.nav_prompt must be one of {NAV_PROMPTS}, got {nav_prompt!r}")
        # Both read <sdroute_cache_path>/<log>/<token>/sdroute_target.gz, so evaluation needs the Scene.
        super().__init__(requires_scene=use_sdroute or nav_prompt == "sdroute_text")
        self._trajectory_sampling = trajectory_sampling
        self.use_sdroute = use_sdroute
        self.sdroute_cache_path = sdroute_cache_path
        self.nav_prompt = nav_prompt
        self.hidden_state_dtype = hidden_state_dtype
        self.vlm_path = vlm_path
        self.checkpoint_path = checkpoint_path
        self.vlm_type = vlm_type
        self.dit_type = dit_type
        self.cache_mode = cache_mode
        self.cache_hidden_state = cache_hidden_state
        self._lr = lr
        self.grpo = grpo
        self.backbone = None
        self.metric_cache_path = metric_cache_path
        self.reference_policy_checkpoint = reference_policy_checkpoint
        self.vlm_size = vlm_size
        self.train_backbone = train_backbone

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        self.device = device
        if not self.cache_hidden_state and not self.cache_mode:
            print("Agent running in 'no-cache' mode. Initializing internal backbone.")
            if not self.vlm_path or not self.vlm_type:
                raise ValueError("In 'no-cache' mode, vlm_path and vlm_type are required.")
            self.backbone = RecogDriveBackbone(
                model_type=self.vlm_type,
                checkpoint_path=self.vlm_path,
                device=device
            )

            for p in self.backbone.parameters():
                p.requires_grad = self.train_backbone

        if self.dit_type == "large":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=1536,sampling_method=sampling_method)
        elif self.dit_type == "small":
            cfg = make_recogdrive_config(self.dit_type, action_dim=3, action_horizon=8, grpo=self.grpo, input_embedding_dim=384,sampling_method=sampling_method)

        cfg.vlm_size = self.vlm_size
        cfg.use_sdroute = use_sdroute
        cfg.drop_command = drop_command
        cfg.sdroute_init_seed = sdroute_init_seed

        if self.grpo:
            cfg.grpo_cfg.metric_cache_path = self.metric_cache_path
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint
            
        self.action_head = ReCogDriveDiffusionPlanner(cfg).cuda()
        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self.checkpoint_path:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
            model_dict = self.state_dict()
            filtered_ckpt, mismatched, ckpt_keys = {}, [], set()
            for k, v in ckpt.items():
                k2 = k[len("agent."):] if k.startswith("agent.") else k
                ckpt_keys.add(k2)
                if k2 not in model_dict:
                    continue
                if v.shape != model_dict[k2].shape:
                    mismatched.append((k2, tuple(v.shape), tuple(model_dict[k2].shape)))
                    continue
                filtered_ckpt[k2] = v
            if mismatched:
                # A shape mismatch means the config disagrees with the checkpoint (e.g. drop_command
                # changes ego_status_encoder's input from 8 to 4); skipping the tensor would leave that
                # layer at random init, so fail instead.
                lines = "\n".join(f"  {k}: checkpoint {c} vs model {m}" for k, c, m in mismatched[:10])
                raise RuntimeError(
                    f"{len(mismatched)} checkpoint tensor(s) do not match this model:\n{lines}\n"
                    f"Check agent.drop_command / agent.dit_type / agent.vlm_size against "
                    f"{self.checkpoint_path}"
                )
            # A checkpoint that carries only part of sdroute_encoder.* was trained with a different
            # route encoder. That is not a shape mismatch, so the check above cannot see it, and
            # load_state_dict(strict=False) would leave the missing tensors at their random init.
            prefix = "action_head.sdroute_encoder."
            want = {k for k in model_dict if k.startswith(prefix)}
            have = {k for k in ckpt_keys if k.startswith(prefix)}
            if want and have and have != want:
                missing, extra = sorted(want - have), sorted(have - want)
                raise RuntimeError(
                    f"checkpoint carries {len(have)} of this model's {len(want)} "
                    f"sdroute_encoder tensors -- a partial route arm, which means the route config "
                    f"does not match the checkpoint.\n"
                    f"  missing from the checkpoint: {missing[:4]}\n"
                    f"  present but unknown here:    {extra[:4]}\n"
                    f"  checkpoint: {self.checkpoint_path}"
                )
            self.load_state_dict(filtered_ckpt, strict=False)

    def get_sensor_config(self) -> SensorConfig:
        # LiDAR is never read (only cam_f0 is); skip loading the point clouds.
        sensor_config = SensorConfig.build_all_sensors(include=[0, 1, 2, 3])
        sensor_config.lidar_pc = False
        return sensor_config

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        builders = [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]
        if self.use_sdroute:
            builders.append(SDRouteTargetBuilder())
        return builders

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [ReCogDriveFeatureBuilder(
            cache_hidden_state=self.cache_hidden_state,
            model_type=self.vlm_type,
            checkpoint_path=self.vlm_path,
            device=self.device,
            cache_mode=self.cache_mode,
            nav_prompt=self.nav_prompt,
            route_cache_path=self.sdroute_cache_path,
            hidden_state_dtype=self.hidden_state_dtype,
        )]

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None) -> Dict[str, torch.Tensor]:
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()

        model_dtype = next(self.action_head.parameters()).dtype

        history_trajectory = features["history_trajectory"].cuda()
        high_command_one_hot = features["high_command_one_hot"].cuda()
        
        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)

        if self.cache_hidden_state:
            last_hidden_state = features["last_hidden_state"].cuda()
        else:
            if self.backbone is None:
                raise RuntimeError("Agent is in 'no-cache' mode, but backbone is not initialized.")
            image_path_tensor = features["image_path_tensor"]
            if image_path_tensor.ndim == 1:
                image_path_tensor = image_path_tensor.unsqueeze(0)
            image_paths = self._decode_paths_from_tensor(image_path_tensor)
            
            pixel_values_list = [load_image(path) for path in image_paths]
            
            num_patches_list = [p.shape[0] for p in pixel_values_list]
            pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda()
            

            if self.nav_prompt == "sdroute_text":
                # set by compute_trajectory from the route cache, with the same function the
                # hidden-state cache used
                if "nav_text" not in features:
                    raise RuntimeError(
                        "agent.nav_prompt='sdroute_text' in online mode needs features['nav_text']; "
                        "only compute_trajectory / compute_trajectory_vis provide it"
                    )
                nav_list = list(features["nav_text"])
            else:
                nav_list = [command_text(c) for c in high_command_one_hot]

            questions = [build_question(history_trajectory[i], nav_list[i])
                         for i in range(high_command_one_hot.shape[0])]

            outputs = self.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list,
                                    max_length=PROMPT_MAX_LENGTH[self.nav_prompt])
            last_hidden_state = outputs.hidden_states[-1]

        status_feature = features["status_feature"].cuda()
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        last_hidden_state = last_hidden_state.to(model_dtype)
        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)

        # SD route rides in targets (cached target); a route model never silently runs without it.
        route_inputs = {}
        if self.use_sdroute:
            if targets is None or "route_centerline" not in targets or "route_centerline_mask" not in targets:
                raise RuntimeError(
                    "use_sdroute=True but targets carry no route_centerline / route_centerline_mask "
                    "(missing sdroute_target.gz in the cache, or an eval path that passes no targets)"
                )
            route = targets["route_centerline"]
            route_mask = targets["route_centerline_mask"]
            if route.ndim == 2:
                route, route_mask = route.unsqueeze(0), route_mask.unsqueeze(0)
            route_inputs = {
                "route_centerline": route.to(device=status_feature.device, dtype=model_dtype),
                "route_centerline_mask": route_mask.to(device=status_feature.device, dtype=torch.bool),
            }

        if self.training and not self.grpo:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype), **route_inputs})
            return self.action_head(last_hidden_state, action_inputs)
        elif self.training and self.grpo:
            action_inputs = BatchFeature(data={"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), "action": targets["trajectory"].to(model_dtype), **route_inputs})
            return self.action_head.forward_grpo(last_hidden_state, action_inputs, tokens_list)
        else:
            action_inputs = BatchFeature({"state": input_state.to(model_dtype), "his_traj": history_trajectory_reshaped.to(model_dtype), "status_feature": status_feature.to(model_dtype), **route_inputs})
            return self.action_head.get_action(last_hidden_state.to(model_dtype), action_inputs)

    def _sdroute_targets(self, scene=None) -> Dict[str, torch.Tensor]:
        """Route for one scene, read from the cache by token (evaluation has no targets dict).

        Read rather than rebuilt: the route must be the tensor training used, so it comes from a
        cache built the same way (scripts/sdroute/run_sdroute_cache.sh, i.e. the shared
        OdysseyZoo/sdroute builder).
        """
        if not self.use_sdroute:
            return {}
        route, mask = self._read_route(scene)
        return {"route_centerline": route.unsqueeze(0), "route_centerline_mask": mask.unsqueeze(0)}

    def _read_route(self, scene=None):
        """route_centerline (P, 5), route_centerline_mask (P,) of one scene from agent.sdroute_cache_path."""
        if scene is None:
            raise RuntimeError(
                "use_sdroute / nav_prompt='sdroute_text' needs the Scene at evaluation time to look up "
                "the route; the caller must honour agent.requires_scene and pass it"
            )
        if not self.sdroute_cache_path:
            raise RuntimeError("use_sdroute / nav_prompt='sdroute_text' but agent.sdroute_cache_path is unset")
        meta = scene.scene_metadata
        return load_cached_route(self.sdroute_cache_path, meta.log_name, meta.initial_token)

    def _add_nav_text(self, features: Dict[str, Any], scene=None) -> None:
        """Online evaluation builds the prompt in forward(); give it the route text for this scene."""
        if self.nav_prompt != "sdroute_text" or self.cache_hidden_state:
            return
        route, mask = self._read_route(scene)
        features["nav_text"] = [sdroute_navigation(route, mask, features["high_command_one_hot"][0])]

    def compute_trajectory(self, agent_input: AgentInput, scene=None) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}
        self._add_nav_text(features, scene)
        targets = self._sdroute_targets(scene)

        with torch.no_grad():
            predictions = self.forward(features, targets or None)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)

        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput, scene=None) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}
        self._add_nav_text(features, scene)
        targets = self._sdroute_targets(scene)

        with torch.no_grad():
            predictions = self.forward(features, targets or None)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)


    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training and self.grpo:
            return predictions
        elif self.training:
            return predictions.loss
        else:
            return torch.nn.functional.l1_loss(predictions["pred_traj"], targets["trajectory"])

    def route_log_stats(self) -> Dict[str, torch.Tensor]:
        """Route-token magnitudes from the last training forward; empty when use_sdroute is off."""
        return getattr(self.action_head, "_route_stats", {}) or {}

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))

        params = list(self.action_head.parameters())
        if self.backbone is not None and self.train_backbone:
            params += list(self.backbone.parameters())

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)
        
        if self.grpo:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=10, warmup_epochs=0)
        else:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0: 
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

def make_recogdrive_config(
    size: str,
    *,
    action_dim: int,
    action_horizon: int,
    input_embedding_dim: int,
    sampling_method: str = 'ddim',
    num_inference_steps: int = 5,
    grpo: bool = False,
    model_dtype: str = "float16",
) -> ReCogDriveDiffusionPlannerConfig:
    """
    A factory function to create a ReCogDriveDiffusionPlannerConfig object.

    This function simplifies configuration by using a size preset ("small",
    "large", "large_new") to define the core DiT architecture, while allowing
    other important planner settings to be specified.

    Args:
        size (str): The size preset for the DiT backbone.
        action_dim (int): The dimension of the action space.
        action_horizon (int): The number of future action steps to predict.
        input_embedding_dim (int): Dimension of the input embeddings to the DiT.
        sampling_method (str): The core training and sampling methodology.
        num_inference_steps (int): Number of steps for inference sampling.
        grpo (bool): If True, enables GRPO-specific logic.
        model_dtype (str): The data type for model computations.

    Returns:
        ReCogDriveDiffusionPlannerConfig: An instantiated and configured planner config object.
    """
    size = size.lower()
    if size == "small":
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16,"output_dim":512}
    elif size == "large":
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16,"output_dim":1536}
    else:
        raise ValueError(f"Unknown model size: {size!r}")

    common_params: Dict[str, any] = {
        "dropout": 0.0,
        "attention_bias": True,
        "norm_eps": 1e-5,
        "interleave_attention": True,
    }
    diffusion_model_cfg.update(common_params)

    config = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg=diffusion_model_cfg,
        action_dim=action_dim,
        action_horizon=action_horizon,
        input_embedding_dim=input_embedding_dim,
        sampling_method=sampling_method,
        num_inference_steps=num_inference_steps,
        grpo=grpo,
        model_dtype=model_dtype,
    )
    
    return config
