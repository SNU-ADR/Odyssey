"""Camera metadata follows the actual rendered optical camera, without GPU imports."""
import ast
from pathlib import Path
import numpy as np
import pytest
from pyquaternion import Quaternion

SOURCE = Path(__file__).resolve().parents[1] / "odyssey/manager/data_manager.py"

def helper():
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_render_camera_calibrations")
    scope = {"np": np, "Quaternion": Quaternion}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope[node.name]

def pose(yaw, xyz):
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.eye(4)
    out[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    out[:3, 3] = xyz
    return out

def test_absent_metadata_preserves_legacy_identity():
    camera = {"cam_intrinsic": np.eye(3), "distortion": np.ones(5)}
    frame = {"cams": {"CAM_F0": camera}}
    helper()(frame, {})
    assert frame["cams"]["CAM_F0"] is camera
    np.testing.assert_array_equal(camera["distortion"], np.ones(5))

def test_nonidentity_lidar_transform_projects_same_camera_point():
    lidar = pose(.37, [1.3, -.4, .8])
    camera = pose(-.21, [.2, .7, 1.4])
    k = np.array([[900., 0, 960], [0, 910, 540], [0, 0, 1]])
    frame = {"lidar2ego": lidar, "cams": {"CAM_F0": {"data_path": "keep.jpg", "sensor2ego_rotation": np.array([1., 0, 0, 0]), "sensor2ego_translation": np.zeros(3)}}}
    item = dict(camera_to_ego=camera, cam_intrinsic=k, distortion=np.zeros(5), projection_semantics="training_undistorted")
    helper()(frame, {"camera_calibrations": {"CAM_F0": item}})
    output = frame["cams"]["CAM_F0"]
    # Native project_to_cam: R.T @ (point_lidar - translation).
    optical = np.array([.5, -.2, 12., 1.])
    point_lidar = np.linalg.solve(lidar, camera @ optical)[:3]
    recovered = output["sensor2lidar_rotation"].T @ (point_lidar - output["sensor2lidar_translation"])
    np.testing.assert_allclose(recovered, optical[:3], atol=1e-12)
    actual = output["cam_intrinsic"] @ recovered
    expected = k @ optical[:3]
    np.testing.assert_allclose(actual / actual[2], expected / expected[2], atol=1e-10)
    assert output["sensor2ego_rotation"].shape == (4,)
    np.testing.assert_allclose(Quaternion(output["sensor2ego_rotation"]).rotation_matrix, camera[:3, :3], atol=1e-12)
    # Native nuScenes convention: scalar first (w, x, y, z).
    np.testing.assert_allclose(output["sensor2ego_rotation"], [np.cos(-.21/2), 0, 0, np.sin(-.21/2)], atol=1e-12)
    np.testing.assert_allclose(output["sensor2ego_translation"], camera[:3, 3], atol=1e-12)
    assert output["data_path"] == "keep.jpg"
    assert output["projection_semantics"] == "training_undistorted"
    output["cam_intrinsic"][0, 0] = 1
    assert k[0, 0] == 900

@pytest.mark.parametrize("field,value", [("distortion", np.ones(5)), ("camera_to_ego", np.ones((4,4))), ("cam_intrinsic", np.full((3,3), np.nan)), ("projection_semantics", "unknown")])
def test_bad_calibration_fails_before_mutating_frame(field, value):
    camera = {"sentinel": 1}
    frame = {"lidar2ego": np.eye(4), "cams": {"CAM_F0": camera}}
    item = dict(camera_to_ego=np.eye(4), cam_intrinsic=np.eye(3), distortion=np.zeros(5), projection_semantics="training_undistorted")
    item[field] = value
    with pytest.raises(ValueError):
        helper()(frame, {"camera_calibrations": {"CAM_F0": item}})
    assert camera == {"sentinel": 1}

def test_data_manager_calls_hook_with_frame_and_render_result():
    tree = ast.parse(SOURCE.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_apply_render_camera_calibrations"]
    assert len(calls) == 1
    assert [n.id for n in calls[0].args] == ["frame_data", "render_results"]
