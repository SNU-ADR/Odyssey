"""The scorer the simulator's MetricManager is configured with (``scorer=odyssey_benchmark.scorer``).

The simulator records a run -- the executed poses, the actors, the signals it drove under -- and
pins it in rollout_trajectory.npz. What it pins and how the run is scored is the benchmark's, so
MetricManager imports none of it: it resolves this module from the ``scorer`` config key (the
benchmark launcher always sets it) and calls the names below. odyssey.manager.metric_manager
.SCORER_API lists them.
"""
from .driving_metrics import (  # noqa: F401
    PinnedInputs,
    apply_termination,
    check_rules_drivable,
    pack_frame,
    pack_map,
    replay,
    rules_from_config,
)
from .sdroute_score import SDRouteMetric  # noqa: F401
from .tlc_metric import capture_tlc_signal_snapshot, pack_tlc  # noqa: F401
