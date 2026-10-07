from typing import Any, List, Dict, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.safedrive.safedrive_config import SafeDrive_Config

from navsim.agents.safedrive.safedrive_model import SafeDrive_Model

from navsim.agents.safedrive.safedrive_callback import SafeDrive_Callback
from navsim.agents.safedrive.safedrive_loss import safedrive_loss

from navsim.agents.safedrive.safedrive_features import SafeDrive_FeatureBuilder, SafeDrive_TargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.safedrive.modules.scheduler import WarmupCosLR
from omegaconf import DictConfig, OmegaConf
import torch.optim as optim
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

import os # Scoring
from pathlib import Path # Scoring

def build_from_configs(obj, cfg: DictConfig, **kwargs):
    if cfg is None:
        return None
    cfg = cfg.copy()
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)
    type = cfg.pop('type')
    return getattr(obj, type)(**cfg, **kwargs)

class SafeDrive_Agent(AbstractAgent):
    """Agent interface for GRAD baseline."""

    def __init__(
        self,
        config: SafeDrive_Config,
        lr: float,
        checkpoint_path: Optional[str] = None,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        """
        Initializes GRAD agent.
        :param config: global config of GRAD agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__(trajectory_sampling)

        self._trajectory_sampling = trajectory_sampling
        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self.include_history = list(range(4 - config.num_input_frames, 4))  # num_input_frames 4 -> [0,1,2,3]

        # NOTE: checkpoint keys are `agent._safedrive_model.*`; renaming this attribute
        # silently breaks weight loading unless the checkpoints are rewritten too.
        self._safedrive_model = SafeDrive_Model(config)
        self.init_from_pretrained()
        # groups built in get_coslr_optimizers(). No-op unless the config enables it.
        self.freeze_perception_modules()
        self.freeze_all_but_markers()

        self.safety_scoring_ray = False

        # Simulator-scored target GT, lazily initialised on first use
        self._scorer_ready = False
        self.train_metric_cache_paths = None
        self.worker = None
        self.get_scores = None

    def _ensure_scorer(self):
        """Lazy setup for the simulator-scored target GT.
        Loads the per-token metric_cache path map (world cache) and, if
        safety_ray_threads > 0, a Ray CPU pool. Local imports keep dataset
        caching free of ray/model deps. Idempotent."""
        if self._scorer_ready:
            return
        from navsim.common.dataloader import MetricCacheLoader
        from .score_module.compute_navsim_score import get_scores
        cache_path = self._config.safety_metric_cache_path
        assert cache_path, ("use_target_scores requires safety_metric_cache_path "
                            "(dir with metadata/*.csv of per-token metric_cache.pkl)")
        self.train_metric_cache_paths = MetricCacheLoader(Path(cache_path)).metric_cache_paths
        if int(self._config.safety_ray_threads) > 0:
            from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
            self.worker = RayDistributedNoTorch(threads_per_node=self._config.safety_ray_threads)
        else:
            self.worker = None
        self.get_scores = get_scores
        self._scorer_ready = True

    def compute_score(self, targets, proposals):
        """proposals: (B, K, T, 3) detached tensor of the model's final refined plan trajs.
        Rolls them out through the PDM simulator/scorer per token -> safety GT.
        Returns (target_scores (B,K,C) float tensor on proposals.device,
                 pair_gt or None, twdac_gt or None)."""
        import numpy as np
        from nuplan.planning.utils.multithreading.worker_utils import worker_map
        self._ensure_scorer()
        pair_nc = bool(getattr(self._config, "pwnc_check", False))
        twdac = bool(getattr(self._config, "twdac_check", False))
        scoring_mode = str(getattr(self._config, "safety_score_mode", "pdms"))
        # the rollout GT feeds the CPU PDM simulator, so fp32 is both safe and accurate.
        data_points = [
            {"token": self.train_metric_cache_paths[t], "poses": p,
             "pair_nc": pair_nc, "twdac": twdac, "scoring_mode": scoring_mode}
            for t, p in zip(targets["scene_frame_token"], proposals.detach().float().cpu().numpy())
        ]
        if self.worker is not None:
            all_res = worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)
        pair_gt = twdac_gt = None
        if pair_nc or twdac:
            # worker returns (scores, extras) per sample when any GT flag is set
            if pair_nc:
                pair_gt = [r[1]["pair_gt"] for r in all_res]
            if twdac:
                twdac_gt = [r[1]["twdac_gt"] for r in all_res]
            all_res = [r[0] for r in all_res]
        target_scores = torch.FloatTensor(np.stack(all_res)).to(proposals.device)
        return target_scores, pair_gt, twdac_gt

    def init_from_pretrained(self):
        """Load a checkpoint, skipping tensors whose shape no longer matches."""
        ckpt_path = getattr(self, "_checkpoint_path", None)
        if not ckpt_path:
            print("[pretrained] no checkpoint_path, initializing from scratch")
            return
        # fail here rather than train for hours from a silently random initialisation
        assert os.path.isfile(ckpt_path), f"checkpoint_path does not exist: {ckpt_path}"

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace('agent.', ''): v for k, v in state_dict.items()}

        own_sd = self.state_dict()
        keep_sd = {}
        skipped = {}  # {key: (ckpt_shape, model_shape or None)}
        for k, v in state_dict.items():
            if k in own_sd:
                if tuple(v.shape) == tuple(own_sd[k].shape):
                    keep_sd[k] = v
                else:
                    skipped[k] = (tuple(v.shape), tuple(own_sd[k].shape))
            else:
                skipped[k] = (tuple(v.shape), None)
        missing_keys, unexpected_keys = self.load_state_dict(keep_sd, strict=False)

        print(f"[pretrained] loaded_from: {ckpt_path}")
        print(f"[pretrained] kept: {len(keep_sd)} tensors")
        if skipped:
            preview = list(skipped.items())[:10]
            print(f"[pretrained] skipped (shape/key mismatch): {len(skipped)}")
            for k, (s_ckpt, s_model) in preview:
                print(f"  - {k}: ckpt{str(s_ckpt)} vs model{str(s_model)}")
            if len(skipped) > 10:
                print(f"  ... and {len(skipped) - 10} more")
        if missing_keys:
            print(f"Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")

    # -----------------------------------------------------------------------
    # Phase2 perception freeze (ported from EAD ead_simplebev_world)
    # -----------------------------------------------------------------------

    def freeze_perception_modules(self):
        """If config.freeze_perception=True, disable grad + eval() for perception
        modules, leaving planning/safety heads trainable. Frozen modules are kept in
        eval() every epoch (see train() override) so BN/running-stat never update.

        Prefixes are top-level SafeDrive_Model attribute names. config.freeze_perception_prefixes
        (a list) overrides the built-in default (exact-module match, so buffers such as
        plan_anchor with requires_grad=False are left untouched)."""
        if not getattr(self._config, "freeze_perception", False):
            return

        # Perception module prefixes only; planning and safety heads stay trainable.
        exclude_prefixes = (
            "ProposalNet_BEV",
            "ins_query_emb",
            "_bev_semantic_head_convnext",
            "future_bev_semantic_head_convnext",
            "ProposalNet_instance",
            "det_cls_branches",
            "det_reg_branches",
            "ins_ref_points",
            "ins_pos_emb_layer",
            "query_upsample",
        )

        cfg_prefixes = getattr(self._config, "freeze_perception_prefixes", None)
        if cfg_prefixes is not None:
            exclude_prefixes = tuple(cfg_prefixes)

        def _is_excluded(name: str) -> bool:
            # exact-module match: "det" prefix must not match "detection_xxx" accidentally.
            return any(name == p or name.startswith(p + ".") for p in exclude_prefixes)

        for name, p in self._safedrive_model.named_parameters():
            if _is_excluded(name):
                p.requires_grad = False

        # set frozen submodules to eval (no BN running-stat drift)
        for name, m in self._safedrive_model.named_modules():
            if name and _is_excluded(name):
                m.eval()
        # Lightning calls .train() every epoch -> remember predicate to re-apply eval
        self._frozen_module_check = _is_excluded

        n_train = sum(p.numel() for p in self._safedrive_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self._safedrive_model.parameters())
        print(f"[Freeze] trainable {n_train / 1e6:.2f}M / total {n_total / 1e6:.2f}M "
              f"(frozen prefixes: {list(exclude_prefixes)})")

    def freeze_all_but_markers(self):
        """A-mode head-only fine-tune: if config.finetune_markers is set, freeze every
        param except those whose name contains one of the raw substrings. Whole model is
        kept in eval(). Overrides freeze_perception."""
        markers = getattr(self._config, "finetune_markers", None)
        if not markers:
            return
        markers = tuple(markers)
        for name, p in self._safedrive_model.named_parameters():
            p.requires_grad = any(mk in name for mk in markers)
        self._safedrive_model.eval()
        self._frozen_module_check = lambda name: True

        n_train = sum(p.numel() for p in self._safedrive_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self._safedrive_model.parameters())
        assert n_train > 0, f"finetune_markers={list(markers)} matched no params"
        print(f"[Freeze-A] trainable {n_train / 1e6:.3f}M / total {n_total / 1e6:.2f}M "
              f"markers={list(markers)}")

    def train(self, mode: bool = True):
        """Keep frozen modules in eval() (Lightning calls .train() every epoch)."""
        super().train(mode)
        frozen_check = getattr(self, "_frozen_module_check", None)
        if mode and frozen_check is not None:
            for name, m in self._safedrive_model.named_modules():
                if name and frozen_check(name):
                    m.eval()
        return self

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        # Only the three cameras SafeDrive_FeatureBuilder actually reads (cam_l0/f0/r0) are loaded:
        # build_all_sensors() would also load cam_l1/l2/r1/r2/b0, which fails when the sensor blobs
        # contain only these three cameras. lidar follows use_lidar, since the lidar feature needs
        # Lidar.lidar_path, which is set only when lidar_pc is requested.
        include = self.include_history  # for history frame
        return SensorConfig(
            cam_f0=include,
            cam_l0=include,
            cam_l1=False,
            cam_l2=False,
            cam_r0=include,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=include if self._config.use_lidar else False,
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [SafeDrive_TargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [SafeDrive_FeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._safedrive_model(features,targets=targets)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
        test_traj_save = False,
        current_epoch = None,
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        # Train-only: roll the model's own final plans through the PDM simulator and put
        # the resulting safety GT into `targets` for the loss to consume.
        if getattr(self._config, "use_target_scores", False) and self.training \
                and self._config.scene_level_safety and not test_traj_save:
            num_plan_layers = self._config.prop_traj_n_layers + self._config.SWNet_num_layers
            plan_key = f"plan_traj_{num_plan_layers - 1}"
            if plan_key in predictions:
                proposals = predictions[plan_key].detach()  # (B, K, T, 3)
                sc, pair_gt, twdac_gt = self.compute_score(targets, proposals)
                targets["safety_target_scores"] = sc
                if pair_gt is not None:
                    targets["pair_collision_gt"] = pair_gt
                if twdac_gt is not None:
                    targets["twdac_gt"] = twdac_gt
        return safedrive_loss(targets, predictions, self._config,
                              test_traj_save=test_traj_save,
                              current_epoch=current_epoch)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return self.get_coslr_optimizers()

    def get_coslr_optimizers(self):
        """AdamW with a warmup-cosine schedule; frozen and low-lr groups are split out."""
        base_lr = self._lr
        lr_down_mode = getattr(self._config, "lr_down", None)  # "image_encoder" | "backbone" | "all_but_heads" | None

        optimizer_cfg = DictConfig(dict(
            type=self._config.optimizer_type,
            lr=base_lr,
            weight_decay=self._config.weight_decay,
        ))
        scheduler_cfg = DictConfig(dict(
            type=self._config.scheduler_type,
            milestones=self._config.lr_steps,
            gamma=0.1,
        ))

        def should_downscale(name: str) -> bool:
            # heads keep the base lr in every mode
            if "scene_level_safety_heads" in name:
                return False

            if lr_down_mode == "image_encoder":
                return "image_encoder" in name
            elif lr_down_mode == "backbone":
                return "ProposalNet_BEV" in name
            elif lr_down_mode == 'detection':
                return "ProposalNet_BEV" in name or "ProposalNet_instance" in name or "det_cls_branches" in name or "det_reg_branches" in name
            elif lr_down_mode == "all_but_heads":
                return True
            else:
                return False  # no down-scaling when unset

        high_lr_params, low_lr_params = [], []
        for name, p in self._safedrive_model.named_parameters():
            if not p.requires_grad:
                continue
            if should_downscale(name):
                low_lr_params.append(p)
            else:
                high_lr_params.append(p)

        param_groups = []
        if lr_down_mode  == "image_encoder":
            if high_lr_params:
                param_groups.append({"params": high_lr_params})
            if low_lr_params:
                param_groups.append({"params": low_lr_params, "lr": base_lr * 0.5})
        else:
            if high_lr_params:
                param_groups.append({"params": high_lr_params})
            if low_lr_params:
                param_groups.append({"params": low_lr_params, "lr": base_lr * self._config.low_lr_scale})

        optimizer = build_from_configs(optim, optimizer_cfg, params=param_groups)

        if self._config.scheduler_type == "cos":
            scheduler = WarmupCosLR(
                optimizer=optimizer,
                lr=base_lr,
                min_lr=1e-6,
                epochs=self._config.epoch,
                warmup_epochs=self._config._warm_epoch,
            )
        else:
            scheduler = None

        if scheduler and "interval" in scheduler_cfg:
            scheduler = {"scheduler": scheduler, "interval": scheduler_cfg["interval"]}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        return [SafeDrive_Callback(self._config)]
