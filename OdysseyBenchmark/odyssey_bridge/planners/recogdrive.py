"""ReCogDrive -- InternVL2-2B VLM + diffusion planner. Camera-only, 1 view.

Repo   OdysseyZoo/models/recogdrive        (github.com/xiaomi-research/recogdrive)
Cfg    recogdrive_baseline_agent.yaml
Ckpt   ckpts/recogdrive_baseline.ckpt                  (planner head)
       ckpts/ReCogDrive-VLM-2B/                        (Stage-1 VLM, 3.9 GB, passed separately)
The code needs Python >= 3.10 (recogdrive_dit.py annotates with `X | Y`): give it its own interpreter.

What makes this backend different from the other nine: it runs a 2B VLM every step. The
base class covers most of that, but four of its defaults do not fit, and each override
below is one of them. No extra selection machinery is layered on either -- see
base.NavsimPlanner's docstring.

  1. images arrive as ndarrays, this model wants PATHS  -> _build_agent_input
  2. the trajectory is under `pred_traj`, not `trajectory` -> _extract_trajectory
  3. sm_75 has no bf16 tensor cores, and fp16 gives NaN  -> _patch_backbone_dtype
  4. it asks for 8 cameras but reads one                -> build

Cost on a Turing (sm_75) GPU in fp32, 1920x1080 -> 9 tiles / 2304 image tokens, replaying
real closed-loop frames:

    InternVL2-2B forward       1.76 s
    full planner step          2.00 s      peak 12.2 GiB (backbone probe)

The runtime places it on the --gpu device.

PLAN_DT stays at the default 0.5: action_horizon=8 and trajectory_sampling.interval_length
is 0.5, i.e. the standard 8 poses x 0.5 s, not a DriveSuprim-style 0.1 s vocabulary.
"""
from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from .base import NavsimPlanner

try:
    from .. import ipc_common as ipc
except ImportError:                      # run as a plain script, not a package
    import ipc_common as ipc             # type: ignore[no-redef]


class ReCogDrivePlanner(NavsimPlanner):
    REPO_ENV = "RECOGDRIVE_REPO"
    #: Fallback directory under models/. This tree names a vendored checkout after its
    #: UPSTREAM REPO, verbatim -- navsim / DrivoR / GTRS / DiffusionDrive all do, and navsim
    #: being lower case is the proof that the name is not prettified. The paper writes
    #: "ReCogDrive"; the repo is xiaomi-research/recogdrive, so this is lower case.
    REPO_DIRNAME = "recogdrive"
    DEFAULT_CFG = "recogdrive_baseline_agent"
    TAG = "recogdrive_rl"
    #: The InternVL preprocessor opens CAM_F0 by path (_build_agent_input, load_image_path).
    CAMERA_FILES = True

    #: Stage-1 VLM. Unlike the planner head it is not the --ckpt, so it is resolved here.
    #: The IL and RL arms share it, so it does not need to be per-arm.
    VLM_SUBDIR = "ckpts/ReCogDrive-VLM-2B"

    _warned_unknown_cmd = False

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        # Freeze this to an absolute path now. base.build() chdir's into the repo, so a
        # relative --repo resolved after that point would miss. __init__ runs before the
        # chdir, which makes this the right place. (the launcher always passes an
        # absolute SEL_REPO, so this only shows up when the planner is driven by hand.)
        self.vlm_path = os.path.abspath(
            os.environ.get("RECOGDRIVE_VLM_PATH")
            or os.path.join(self.repo, self.VLM_SUBDIR))

    # ------------------------------------------------------------------ build
    def _build_agent(self) -> None:
        """Inject the VLM path alongside the checkpoint, and turn the feature cache off.

        The yaml ships ``cache_hidden_state: True``. That is the offline path: it reads
        InternVL hidden states dumped ahead of time, which is how the repo trains and
        evaluates on navtest. Closed loop has no such cache, so False makes the agent hold
        the backbone itself and run it every step -- the same setting the upstream
        evaluation script uses (scripts/evaluation/run_recogdrive_agent_pdm_score_
        evaluation_2b.sh passes agent.cache_hidden_state=False).

        ``cache_mode`` must be False too. If either is True the feature builder constructs
        a SECOND backbone (recogdrive_features.py) and the VLM is resident twice.
        """
        if not os.path.isdir(self.vlm_path):
            raise SystemExit(
                f"[{self.TAG}] VLM directory not found: {self.vlm_path!r}\n"
                f"  Override with RECOGDRIVE_VLM_PATH, or place it at "
                f"{self.repo}/{self.VLM_SUBDIR}\n"
                f"  (huggingface.co/owl10/ReCogDrive-VLM-2B, 3.9 GB)")

        self._patch_backbone_dtype()

        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        agent_cfg_dir = os.path.join(
            self.repo, "navsim/planning/script/config/common/agent")
        with initialize_config_dir(version_base=None, config_dir=agent_cfg_dir):
            cfg = compose(config_name=self.cfg_name, overrides=self.overrides)
        if self.overrides:
            print(f"[{self.TAG}] config overrides: {self.overrides}", flush=True)

        self.agent = instantiate(
            cfg,
            checkpoint_path=self.ckpt_path,
            vlm_path=self.vlm_path,
            cache_hidden_state=False,
            cache_mode=False,
        )
        self.agent.initialize()

    @staticmethod
    def _patch_backbone_dtype() -> None:
        """Drop the VLM to float32 on cards without bf16 tensor cores. Also disable flash-attn.

        recogdrive_backbone.py hardcodes ``torch_dtype=torch.bfloat16`` and
        ``use_flash_attn=True``. Turing (sm_75) cards support neither.

        **fp16 is NOT the answer here -- it returns NaN.** On a Turing card, same image, same
        weights, only the dtype changed:

            dtype    vision feature      last_hidden_state       time      VRAM
            fp16     fine (absmax 11.9)  **all NaN**             586 ms   8.42 GiB
            bf16     fine                fine (absmax 125.5)    2442 ms   8.77 GiB
            fp32     fine                fine (absmax 127.4)    1762 ms  12.21 GiB

        The vision encoder is fine in fp16; the LLM half overflows. It is Qwen2, and
        transformers' eager path builds the attention logits BEFORE scaling them:

            modeling_qwen2.py:288
                attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)

        so the unscaled dot product has to survive fp16 first. Hooks put k_proj's absmax at
        319 and q_proj's at 105, and head_dim is 128 -- the product reaches ~4.3e6 against
        fp16's 65504 ceiling. inf -> softmax -> NaN, propagated through all 28 layers. (The
        softmax on line 305 already upcasts to fp32, but by then the damage is done.) bf16
        survives because it trades mantissa for fp32's exponent range, which is exactly what
        the checkpoint was trained in.

        **fp32 is also FASTER than bf16 here** -- 1762 vs 2442 ms. Turing has no bf16 tensor
        cores, so PyTorch emulates it on a slow path (GEMM 2048^3: 6.6 TFLOPS bf16 vs
        12.9 fp32). The cost is VRAM: 8.8 -> 12.2 GiB.

        flash-attn needs no help -- InternVL's remote code checks ``has_flash_attn`` and
        falls back to eager on its own (modeling_internvl_chat.py). It is disabled
        explicitly anyway, so that installing flash-attn into this env later cannot break
        sm_75 at runtime.

        Nothing downstream disagrees: the diffusion head is fp32 already and agent.forward
        casts the hidden state to the head's dtype (recogdrive_agent.py).

        On a card WITH bf16 tensor cores (sm_80+) this does nothing, so the file does not
        quietly become a different model on a different GPU generation.
        """
        import torch

        if torch.cuda.is_bf16_supported():
            return

        import navsim.agents.recogdrive.recogdrive_backbone as bb

        if getattr(bb, "_odyssey_fp32_shim", False):
            return
        real = bb.AutoModel

        class _FP32AutoModel:
            """Module-local shim, so transformers.AutoModel is not patched globally."""

            @staticmethod
            def from_pretrained(path, **kw):
                kw["torch_dtype"] = torch.float32     # fp16 gives NaN -- see the docstring
                kw["use_flash_attn"] = False
                return real.from_pretrained(path, **kw)

        bb.AutoModel = _FP32AutoModel
        bb._odyssey_fp32_shim = True
        cap = torch.cuda.get_device_capability()
        print(f"[recogdrive] sm_{cap[0]}{cap[1]}: no bf16 tensor cores -> float32, "
              f"flash-attn off", flush=True)

    def build(self) -> None:
        super().build()

        # This model reads exactly one view: cam_f0 of the newest frame
        # (recogdrive_features.py, recogdrive_agent.py). Its get_sensor_config,
        # though, asks for all 8 cameras AND lidar across 4 frames
        # (recogdrive_agent.py, build_all_sensors(include=[0,1,2,3])).
        #
        # In a rollout narrowed by ODYSSEY_RENDER_CAMS=cam_f0, the seven unrendered views carry
        # an EMPTY data_path (data_manager.py). Leaving them in the request would make
        # navsim treat sensor_blobs_root / "" -- a directory -- as an image file. Ask for
        # what is actually read instead.
        #
        # ipc.patch_lidar_for_camera_only already neutralises the lidar; dropping it here
        # as well keeps "what this model consumes" readable in one place.
        from navsim.common.dataclasses import SensorConfig

        self._sensor_config = SensorConfig(
            cam_f0=[ipc.NUM_HISTORY_FRAMES - 1],   # newest frame only
            cam_l0=False, cam_l1=False, cam_l2=False,
            cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False,
            lidar_pc=False,
        )
        print(f"[{self.TAG}] sensors: cam_f0 @ frame {ipc.NUM_HISTORY_FRAMES - 1} only "
              f"(VLM {os.path.basename(self.vlm_path)})", flush=True)

    # ------------------------------------------------------------------ infer
    def _build_agent_input(self, scene_list: List[dict]):
        """Hand the images over as PATHS, not decoded arrays. The one place base differs.

        base does not pass ``load_image_path``, so Camera.image ends up as
        ``np.array(Image.open(path))`` (navsim/common/dataclasses.py). The other eight
        backends consume that array. ReCogDrive reads the same field as a FILE PATH and
        gives it to the InternVL preprocessor:

            recogdrive_features.py    image_path = str(cameras[-1].cam_f0.image)
            recogdrive_agent.py       pixel_values = load_image(path)   # Image.open

        which is why the upstream evaluation script runs with load_image_path=True
        (navsim/planning/script/run_pdm_score_recogdrive.py).

        Without this, ``str(ndarray)`` becomes a path-shaped string like "[[[ 34  56 ..."
        and the first step dies on FileNotFoundError. Loud, but every rollout dies.
        """
        from navsim.common.dataclasses import AgentInput

        ipc.patch_lidar_for_camera_only()
        return AgentInput.from_scene_dict_list(
            scene_list,
            self.sensor_blobs_root,
            num_history_frames=ipc.NUM_HISTORY_FRAMES,
            sensor_config=self._sensor_config,
            load_image_path=True,
        )

    def _build_features(self, agent_input, scene_list: List[dict]) -> Dict:
        """Clamp an 'unknown' driving command to 'straight'.

        The simulator's one-hot is 4-wide -- ``[left, straight, right, unknown]`` (ipc_common.py)
        -- and the log stores `unknown` where it could not resolve a command, so unknown can
        reach us.

        ReCogDrive's no-cache path indexes a 3-element list with an argmax over all four:

            recogdrive_agent.py       navigation_commands = ['turn left', 'go straight', 'turn right']
            recogdrive_agent.py       command_str_list = [navigation_commands[idx.item()] for idx in ...]

        so index 3 is an IndexError that kills the rollout. Unknown is effectively absent
        from the navsim training distribution too, i.e. an input the model never saw.
        Clamping is better than dying -- but it is LOGGED, because silently rewriting the
        model's only routing signal would make the score unexplainable later.

        status_feature is built from the same one-hot (recogdrive_features.py), so
        fixing it here keeps both paths consistent.
        """
        for status in agent_input.ego_statuses:
            cmd = np.asarray(status.driving_command)
            if cmd.shape[-1] > 3 and int(np.argmax(cmd)) == 3:
                status.driving_command = np.array([0, 1, 0, 0], dtype=cmd.dtype)
                if not ReCogDrivePlanner._warned_unknown_cmd:
                    print(f"[{self.TAG}] driving_command 'unknown' -> 'straight' "
                          f"(this model knows only three); not logged again", flush=True)
                    ReCogDrivePlanner._warned_unknown_cmd = True
        return super()._build_features(agent_input, scene_list)

    def _extract_trajectory(self, out: Dict) -> np.ndarray:
        """get_action returns ``BatchFeature({"pred_traj": (B, 8, 3)})``, not "trajectory".

        recogdrive_diffusion_planner.py. Rename the key and let base do the shape
        checking and the batch-dim drop.
        """
        return super()._extract_trajectory({"trajectory": out["pred_traj"]})
