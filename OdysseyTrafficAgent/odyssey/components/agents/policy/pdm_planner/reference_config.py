"""Reference PDM-Closed parameters from the public tuPlan Garage release.

Keep these values in one place: per-module copies in the driving policy and the scoring
managers would silently drift away from the CoRL 2023 implementation.  Environment adapters may change how data reaches PDM, but should not
change the planner's proposal family without an explicit experiment flag.

References:
https://github.com/autonomousvision/tuplan_garage/blob/main/tuplan_garage/planning/script/config/simulation/planner/pdm_closed_planner.yaml
https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/script/config/simulation/ego_controller/tracker/lqr_tracker.yaml
"""

from odyssey.components.agents.policy.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)


TRAJECTORY_NUM_POSES = 80
PROPOSAL_NUM_POSES = 40
SAMPLE_INTERVAL = 0.1

SPEED_LIMIT_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
FALLBACK_TARGET_VELOCITY = 15.0
MIN_GAP_TO_LEAD_AGENT = 1.0
HEADWAY_TIME = 1.5
ACCEL_MAX = 1.5
DECEL_MAX = 3.0
LATERAL_OFFSETS = (-1.0, 1.0)
MAP_RADIUS = 50.0

# nuPlan's LQR is the low-level controller used to realize planner trajectories. These are not
# proposal parameters, but keeping the reference controller beside the reference planner avoids
# silently evaluating PDM with the repository's unrelated learned-policy tuning.
LQR_Q_LONGITUDINAL = (10.0,)
LQR_R_LONGITUDINAL = (1.0,)
LQR_Q_LATERAL = (1.0, 10.0, 0.0)
LQR_R_LATERAL = (1.0,)


def build_reference_idm_policy(*, speed_scale: float = 1.0) -> BatchIDMPolicy:
    """Construct the PDM longitudinal proposal batch, optionally slowed for an experiment.

    The default is exactly the public PDM-Closed proposal family.  ``speed_scale`` belongs to
    the ego planner only: callers used for scoring keep the default, and background nuPlan IDM
    has a separate policy/configuration path.
    """
    speed_scale = float(speed_scale)
    if not 0.0 < speed_scale <= 1.0:
        raise ValueError(f"PDM speed_scale must be in (0, 1], got {speed_scale}")
    return BatchIDMPolicy(
        speed_limit_fraction=[fraction * speed_scale for fraction in SPEED_LIMIT_FRACTIONS],
        fallback_target_velocity=FALLBACK_TARGET_VELOCITY * speed_scale,
        min_gap_to_lead_agent=MIN_GAP_TO_LEAD_AGENT,
        headway_time=HEADWAY_TIME,
        accel_max=ACCEL_MAX,
        decel_max=DECEL_MAX,
    )
