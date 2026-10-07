import pickle
import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, Scene, SensorConfig


# Mapping from cache component key to weight param name
COMPONENT_TO_WEIGHT = {
    "plan_comp": "imi_test_weight",
    "NC_comp": "NC_test_weight",
    "DAC_comp": "DAC_test_weight",
    "EP_comp": "EP_test_weight",
    "TTC_comp": "TTC_test_weight",
    "C_comp": "C_test_weight",
    "pdm_comp": "pdm_score_test_weight",
    "DDC_comp": "DDC_test_weight",
    "TLC_comp": "TLC_test_weight",
    "LK_comp": "LK_test_weight",
    "HC_comp": "HC_test_weight",
    "twdac_comp": "TwDAC_test_weight",
    "pair_NC_comp": "PwNC_test_weight",
}


class LoadScoreCacheAgent(AbstractAgent):
    """Agent that loads score cache and applies weighted sum to select trajectory."""

    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        score_cache_path: str = None,
        imi_test_weight: float = 1.0,
        NC_test_weight: float = 0.0,
        DAC_test_weight: float = 0.0,
        EP_test_weight: float = 0.0,
        TTC_test_weight: float = 0.0,
        C_test_weight: float = 0.0,
        pdm_score_test_weight: float = 0.0,
        DDC_test_weight: float = 0.0,
        TLC_test_weight: float = 0.0,
        LK_test_weight: float = 0.0,
        HC_test_weight: float = 0.0,
        PwNC_test_weight: float = 0.0,
        TwDAC_test_weight: float = 0.0,
        **kwargs,
    ):
        self._trajectory_sampling = trajectory_sampling
        self._score_cache_path = score_cache_path
        self._weights = {
            "imi_test_weight": imi_test_weight,
            "NC_test_weight": NC_test_weight,
            "DAC_test_weight": DAC_test_weight,
            "EP_test_weight": EP_test_weight,
            "TTC_test_weight": TTC_test_weight,
            "C_test_weight": C_test_weight,
            "pdm_score_test_weight": pdm_score_test_weight,
            "DDC_test_weight": DDC_test_weight,
            "TLC_test_weight": TLC_test_weight,
            "LK_test_weight": LK_test_weight,
            "HC_test_weight": HC_test_weight,
            "PwNC_test_weight": PwNC_test_weight,
            "TwDAC_test_weight": TwDAC_test_weight,
        }
        self._score_cache = None

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self._score_cache_path is not None and self._score_cache is None:
            with open(self._score_cache_path, "rb") as f:
                self._score_cache = pickle.load(f)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        """Computes trajectory by applying weighted sum to cached score components."""
        self.initialize()
        token = scene.frames[scene.scene_metadata.num_history_frames - 1].token

        if "data" in self._score_cache:
            data = self._score_cache["data"]
        else:
            data = self._score_cache

        assert token in data, f"Token {token} not found in score cache."

        entry = data[token]
        plan_reg = entry["plan_reg"]  # [K, num_poses, 3]

        # Weighted sum over all available score components
        score = np.zeros(plan_reg.shape[0], dtype=np.float64)
        for comp_key, weight_key in COMPONENT_TO_WEIGHT.items():
            if comp_key not in entry:
                continue
            w = self._weights.get(weight_key, 0.0)
            if w == 0.0:
                continue
            comp = np.asarray(entry[comp_key], dtype=np.float64)
            score += w * comp

        # Fallback if all scores are -inf
        if np.all(np.isinf(score) & (score < 0)):
            plan_comp = np.asarray(entry["plan_comp"], dtype=np.float64)
            mode_idx = int(np.argmax(plan_comp))
        else:
            mode_idx = int(np.argmax(score))

        poses = plan_reg[mode_idx]
        return Trajectory(poses)
