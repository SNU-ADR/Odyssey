"""Every physical and protocol constant in this repository, in one file.

The rule this file exists to enforce:

    A number that is not in here is a parameter. It comes from outside -- config,
    CLI, or the scene -- and the code that uses it must be able to say where it
    came from.

Constants that are re-typed per file drift silently: every copy is a place where a rollout
can go wrong while each individual file still looks correct.

So: no bare literals for these values anywhere else, and no per-module copies. Import
from here. The trees each have one thin adapter that puts this file on the path --
`odyssey.utils.cadence` for the simulator, `ipc_common` for the bridge -- so most code
imports from its own tree and this file stays the single declaration.


THE RULE
========
1. Do not compute a value that something outside already decided.
2. If you truly must compute one, compute it from what is in this file.
3. There is no third case. Anything else is a bug generator.

The shape to watch for is a value re-derived from a PROXY: a ratio, a length, a flag
that happens to correlate with what you want. The proxy holds under an assumption
nobody wrote down, and when it stops holding the code keeps running and returns a
plausible but wrong number.

Consumers read; they do not re-derive. `odyssey.utils.cadence.resolve_cadence` is
the only place the step is decided, and `scene["cadence"]` carries the answer to every
consumer. If it is missing, fail -- do not guess.


TIME
====
Four clocks, and only one of them is a knob.

    controller == plant                      CONTROL_DT        constant
    rendering  == replanning == outer step   the knob          rollout_dt, in SIM_DT_CHOICES
    waypoint spacing on the wire             PLAN_FILE_DT      constant
    planner pose spacing / history spacing   PLANNER_POSE_DT   fixed by training
    navsim/PDM scoring grid                  SCORE_DT          fixed by the protocol

`rollout_dt` is the only free choice. The plant runs `rollout_dt / CONTROL_DT` sub-steps
per outer step; the scorer grades every `SCORE_DT / rollout_dt`-th step. Anything else
claiming to be one of these is a duplicate.
"""

# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #

# The controller and the plant tick together, always, at this period. The LQR gains
# (tracking_horizon=10) are tuned for it. A 0.5 s outer step runs five of these
# sub-steps; a 0.1 s outer step runs one.
CONTROL_DT = 0.1

# Row spacing of the planner's trajectory .npy on the wire between the two processes.
# Equal to CONTROL_DT by construction -- the bridge upsamples the model's poses onto
# exactly this grid -- but it is a separate contract: change one without the other and
# the file still parses and still loads. Hence a separate name.
PLAN_FILE_DT = 0.1

# Spacing of the planner model's own output poses, and of the history frames fed to it.
# Fixed by training. Not ours to choose.
PLANNER_POSE_DT = 0.5

# History frames the planner model is fed. A tensor SHAPE the network was trained with, not
# a duration -- the frames are PLANNER_POSE_DT apart, so at a finer outer step the buffer has
# to reach back (num_history_frames - 1) * stride + 1 steps to cover the same span.
NUM_HISTORY_FRAMES = 4

# navsim/PDM score on this grid no matter how finely the world was stepped.
# Fixed by the evaluation protocol. Not ours to choose.
SCORE_DT = 0.5

# nuPlan's lidar period. A scenario's `sample_rate` is a COUNT of these, not a frequency:
# sample_rate=2 means 0.1 s frames, sample_rate=10 means 0.5 s frames. Do not read it as Hz.
NUPLAN_LIDAR_DT = 0.05

# The outer step values the simulator supports. Rendering and replanning both happen once
# per outer step, so this is also the set of supported render/replan periods. Anything
# else either does not divide CONTROL_DT or does not divide into SCORE_DT.
SIM_DT_CHOICES = (0.1, 0.5)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

# Chord length used to read a heading off a trajectory that did not come with one, in
# metres. Only a fallback: a model that emits poses without an orientation is under-
# specifying its own plan, and the recovered heading is an approximation of what it meant.
# Measured against the models that DO emit one, a 1 m chord recovers it to ~1 degree median
# and ~4 degrees at p90.
# It only ever produces a heading; it never decides which waypoints the position is
# interpolated through.
HEADING_CHORD_M = 1.0


# --------------------------------------------------------------------------- #
# Episode termination and arrival -- the benchmark contract (docs/benchmark.md)
# --------------------------------------------------------------------------- #

# A rollout ends as `route_deviation` (P_SD = 0) once the ego is this far from the scored
# SD route, in metres.
SD_ROUTE_MAX_DIST_M = 30.0

# A rollout ends as `destination_arrival` once it has covered this share of the scored SD
# route AND is within this distance of its end. The scorer also drops the trajectory inside
# that radius from route matching (odyssey_benchmark.sdroute_score.arrival_zone_keep), so both read it here.
SD_GOAL_PROGRESS_RATIO = 0.99
SD_GOAL_END_DIST_M = 10.0

# The same pair for the fallback goal, the logged ego's end, when a scene has no usable SD
# route.
GT_GOAL_PROGRESS_RATIO = 0.99
GT_GOAL_END_DIST_M = 5.0
