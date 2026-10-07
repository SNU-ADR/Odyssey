"""Per-proposal PDM sub-score GT, computed on the fly for safety supervision.

Runs PDMSimulator + PDMScorer on the model's own predicted trajectories during the
training loss pass. Only the per-token `metric_cache` (observation / map / centerline)
is read from disk; the score itself is computed live."""
import lzma
import pickle

import numpy as np

from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)

from .train_pdm_scorer import PDMScorerConfig, PDMScorer
from .token_id import token_to_id

proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
simulator = PDMSimulator(proposal_sampling)
config = PDMScorerConfig()
scorer = PDMScorer(proposal_sampling, config)


def get_scores(args):
    return [get_sub_score(a["token"], a["poses"], a.get("pair_nc", False), a.get("twdac", False),
                          a.get("scoring_mode", "pdms"))
            for a in args]


def get_sub_score(metric_cache_path, poses, return_pair_collisions=False, return_twdac=False,
                  scoring_mode="pdms"):
    """poses: (K, num_poses, 3). scoring_mode "pdms" -> (K, 7) = [NC, DAC, EP, TTC, comfort, DDC,
    final]; "epdms" -> (K, 9) = [NC, DAC, EP, TTC, comfort, DDC, TLC, LK, final] (final via v2
    EPDMS aggregation with DDC/TLC as multipliers). return_pair_collisions/return_twdac add the
    pwnc/twdac GT extras identically in both modes."""
    with lzma.open(metric_cache_path, "rb") as f:
        metric_cache = pickle.load(f)

    initial_ego_state = metric_cache.ego_state

    trajectory_states = []
    for model_trajectory in poses:
        pred_trajectory = transform_trajectory(Trajectory(model_trajectory), initial_ego_state)
        pred_states = get_trajectory_as_array(pred_trajectory, simulator.proposal_sampling,
                                              initial_ego_state.time_point)
        trajectory_states.append(pred_states)

    trajectory_states = np.stack(trajectory_states, axis=0)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)

    final_scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        getattr(metric_cache, "pdm_progress", 0.0),  # eval-style caches lack it; epdms EP doesn't use it
        scoring_mode=scoring_mode,
    )

    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, :]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, :]
    driving_direction_compliance = scorer._weighted_metrics[WeightedMetricIndex.DRIVING_DIRECTION, :]

    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, :]
    time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, :]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.COMFORTABLE, :]

    base_cols = [no_at_fault_collisions, drivable_area_compliance,
                 ego_progress, time_to_collision_within_bound, comfort,
                 driving_direction_compliance]  # cols 0,1,3,4,5 shared; EP (col 2) v1-coupled (pdms) / v2-decoupled (epdms)
    if scoring_mode == "epdms":
        scores = np.stack(base_cols + [scorer._tlc_scores, scorer._lk_scores, final_scores],
                          axis=-1)  # (K, 9): +TLC +LK
    else:
        scores = np.stack(base_cols + [final_scores], axis=-1)  # (K, 7)
    if not return_pair_collisions and not return_twdac:
        return scores

    # both GTs bin 0.1s (40-pose) sim indices into the model's num_pose poses (t0 not attributable)
    extras = {}
    num_pose = poses.shape[1]
    ratio = proposal_sampling.num_poses // num_pose

    if return_pair_collisions:
        # bin at-fault (token, 0.1s time_idx) hits into the K input poses
        pair_gt = []
        for k in range(poses.shape[0]):
            d = {}
            for token, t in scorer.proposal_fault_collision_times.get(k, ()):
                if t == 0:
                    continue
                d.setdefault(token_to_id(token), set()).add(min((t - 1) // ratio, num_pose - 1))
            pair_gt.append({tid: sorted(v) for tid, v in d.items()})
        extras["pair_gt"] = pair_gt

    if return_twdac:
        # per-proposal off-road pose indices from the pre-collapse drivable-area boolean
        # (scorer.off_road (K, 41); drop t0 via [:, 1:] -> 0-based step t maps to min(t//ratio, T-1))
        off = scorer.off_road[:, 1:]                                # (K, 40) bool
        twdac_gt = []
        for k in range(off.shape[0]):
            steps = np.nonzero(off[k])[0]                           # off-road 0.1s step indices
            pose_idcs = sorted({min(int(t) // ratio, num_pose - 1) for t in steps})
            twdac_gt.append(pose_idcs)                              # OFF-ROAD pose indices
        extras["twdac_gt"] = twdac_gt

    return scores, extras
