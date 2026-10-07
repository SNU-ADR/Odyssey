"""Actor-row cleanup for saved scoring inputs. Uses only `math`.

Kept separate so that driving_metrics.replay and the traffic_efficiency.py CLI select rows with
**the same function**. driving_metrics imports shapely, so a numpy-only CLI cannot import it --
yet if the CLI kept its own copy, two places would decide which rows are real.
"""
import math


#: A move larger than this in one scoring step is not physical motion (10 m in 0.1 s = 360 km/h).
UNSET_POSE_JUMP_M = 10.0


def drop_unset_pose_frames(actor_frames):
    """Drop the last row of actors that jump to the start point without a pose. -> (frames, dropped)

    An actor whose source (log) has ended appears for one frame, just before it disappears, at
    local coordinates (0, 0). The code that builds the scoring input maps that back to
    initial_ego_center, so all such actors pile up where the rollout started. Early in the run
    the ego is still there, so these are recorded as **contacts with actors that are not on
    screen**.

    In a reactive run these rows can number in the hundreds, **all at a single coordinate**,
    and some are counted as contacts.

    The producer side (odyssey_to_pdm_utils) no longer emits them. This function exists to
    **fix already-recorded runs by re-scoring**. The record does not keep the origin, so rows are
    selected by signature rather than by absolute position: the token's last row, which jumped
    more than 10 m from the previous row and has velocity exactly 0. No real actor moves 10 m in
    0.1 s, and certainly not while also reporting zero velocity.
    """
    seen = {}
    for i, frame in enumerate(actor_frames):
        for row in frame:
            seen.setdefault(row[0], []).append((i, row))
    drop = set()
    for token, rows in seen.items():
        if len(rows) < 2:
            continue
        (i, last), (_, prev) = rows[-1], rows[-2]
        if float(last[8]) or float(last[9]):
            continue
        if math.hypot(last[2] - prev[2], last[3] - prev[3]) <= UNSET_POSE_JUMP_M:
            continue
        drop.add((i, token))
    if not drop:
        return actor_frames, 0
    out = [[row for row in frame if (i, row[0]) not in drop]
           for i, frame in enumerate(actor_frames)]
    return out, len(drop)
