"""What the planner receives of a simulator frame: no ground truth, everything else unchanged."""
import numpy as np

from odyssey_runtime.session import PRIVILEGED_FIELDS, planner_view


def test_planner_view_drops_ground_truth_and_keeps_the_agent_input():
    frame = dict(token="t", frame_idx=3, timestamp=1, cams={"CAM_F0": {}}, lidar_path=None,
                 ego2global_translation=np.zeros(3), ego2global_rotation=np.array([1., 0, 0, 0]),
                 ego_dynamic_state=np.zeros(4), driving_command=np.array([0, 1, 0, 0]), log_token="l",
                 anns={"gt_boxes": np.zeros((2, 7))}, gt_boxes=np.zeros((2, 7)), gt_fut_bbox_sdc_global=np.zeros(1),
                 roadblock_ids=["1"], route_roadblock_ids=["1"], traffic_lights=[("c", True)],
                 source_rows={"a": 1}, signal_rows={"c": 1})
    view = planner_view(frame)
    assert not set(view) & PRIVILEGED_FIELDS and not any(k.startswith("gt_") for k in view)
    assert set(view) == {"token", "frame_idx", "timestamp", "cams", "lidar_path", "ego2global_translation",
                         "ego2global_rotation", "ego_dynamic_state", "driving_command", "log_token"}
    assert all(view[k] is frame[k] for k in view)
    assert "anns" in frame                       # the simulator's own frame is not changed
