"""One synthetic planner step inside the model's own process, for `odyssey_runtime check`.

`check` starts `probe_worker` instead of PlannerWorker. The model is built exactly as in a run;
then the worker's own `handle()` is called with one observation:

- camera images: a fixed synthetic picture per declared camera, 1920x1080, sent as JPEG bytes
  like the simulator sends them, with the real rig's calibration (undistorted pinhole, K below);
- ego: at rest pose (heading +x), driving at 5 m/s;
- driving command: straight; SD route: a straight synthetic route, both in the runtime's format.

That step reports the plan and the feature tensors the model received. Then it varies each
navigation input the model is fed (left command; a route that turns left) and compares plans:
a plan that is bit-identical when the route changes means the model never reads the route
where the runtime puts it (e.g. `sd_route: features` declared for a model that reads targets).
"""
import os
import random
import time
import traceback

import numpy as np

# The camera rig the simulator renders (sensor->lidar, which equals sensor->ego here).
RIG = {
    "CAM_F0": dict(rotation=[[-0.00317, -0.04173, 0.99912], [-0.99997, -0.00686, -0.00346], [0.007, -0.99911, -0.04171]],
                   translation=[1.65899, -0.00869, 1.51035], quaternion=[-0.48689, 0.51122, -0.50942, 0.49202]),
    "CAM_L0": dict(rotation=[[0.82292, -0.00125, 0.56816], [-0.56784, -0.03514, 0.82239], [0.01894, -0.99938, -0.02962]],
                   translation=[1.66386, 0.17384, 1.51295], quaternion=[-0.66298, 0.68696, -0.2071, 0.21366]),
    "CAM_R0": dict(rotation=[[-0.8246, -0.00028, 0.56571], [-0.5656, 0.02047, -0.82443], [-0.01136, -0.99979, -0.01704]],
                   translation=[1.72218, -0.16469, 1.5108], quaternion=[0.21144, -0.20734, 0.6823, -0.66841]),
    "CAM_L1": dict(rotation=[[0.93168, 0.01786, -0.36285], [0.36289, 0.001, 0.93183], [0.017, -0.99984, -0.00555]],
                   translation=[1.28637, 0.65285, 1.39716], quaternion=[-0.6941, 0.69574, 0.13681, -0.12427]),
    "CAM_R1": dict(rotation=[[-0.92708, 0.00653, -0.37481], [0.37486, 0.02215, -0.92682], [0.00225, -0.99973, -0.02298]],
                   translation=[1.26764, -0.64706, 1.39652], quaternion=[-0.13425, 0.13579, 0.70218, -0.68592]),
    "CAM_L2": dict(rotation=[[0.63178, 0.01442, -0.77501], [0.77515, -0.01412, 0.63162], [-0.00184, -0.9998, -0.0201]],
                   translation=[-0.48609, 0.54334, 1.38141], quaternion=[-0.63197, 0.64537, 0.30586, -0.30093]),
    "CAM_R2": dict(rotation=[[-0.62446, 0.02808, -0.78055], [0.78072, -0.00674, -0.62484], [-0.0228, -0.99958, -0.01771]],
                   translation=[-0.53277, -0.56751, 1.37286], quaternion=[-0.29626, 0.31622, 0.63942, -0.63512]),
    "CAM_B0": dict(rotation=[[-0.005, 0.00201, -0.99999], [0.99995, -0.00845, -0.00502], [-0.00846, -0.99996, -0.00196]],
                   translation=[-0.53487, 0.02905, 1.46546], quaternion=[-0.49613, 0.50135, 0.49963, -0.50286]),
}
INTRINSICS = [[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]]
IMAGE_HW = (1080, 1920)
ORIGIN = (1000.0, 2000.0)         # global ego position; heading 0 (+x)
SPEED = 5.0                       # m/s
STRAIGHT, LEFT = [0, 1, 0, 0], [1, 0, 0, 0]


def route_polyline(turn):
    """Global (x, y) at 1 m spacing from 30 m behind the ego: straight, or a left turn after 10 m."""
    x0, y0 = ORIGIN
    if not turn:
        xs = np.arange(-30.0, 301.0)
        return np.column_stack([x0 + xs, np.full_like(xs, y0)])
    radius = 25.0
    lead = np.column_stack([x0 + np.arange(-30.0, 10.0), np.full(40, y0)])
    a = np.linspace(0.0, np.pi / 2, int(radius * np.pi / 2), endpoint=False)
    arc = np.column_stack([x0 + 10.0 + radius * np.sin(a), y0 + radius * (1 - np.cos(a))])
    tail = np.column_stack([np.full(260, x0 + 10.0 + radius), y0 + radius + np.arange(260.0)])
    return np.vstack([lead, arc, tail])


def write_route(path, turn):
    """A route file in the format of the scenes' route.npz (StaticRouteCenterline)."""
    xy = route_polyline(turn)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    np.savez(path, route_xy=xy, route_s=s, verdict=np.array("probe"), src_tokens=np.array(["probe"]),
             scene_name=np.array("odyssey_probe"))
    return str(path)


def synthetic_image(name):
    """A smooth sky/road picture with mild noise, different per camera; uint8 HWC BGR."""
    h, w = IMAGE_HW
    rows = np.linspace(0.0, 1.0, h)[:, None, None]
    image = np.where(rows < 0.45, 170 - 60 * rows, 95 - 30 * (rows - 0.45)) * np.ones((1, w, 3))
    noise = np.random.RandomState(sorted(RIG).index(name)).randint(-8, 9, (h, w, 3))
    return np.clip(image + noise, 0, 255).astype(np.uint8)


def frame(command, cameras):
    """The planner-side observation dict the simulator sends (no privileged annotations)."""
    x, y = ORIGIN
    ego2global = np.eye(4)
    ego2global[:2, 3] = (x, y)
    cams = {}
    for name, rig in RIG.items():
        to_ego = np.eye(4)
        to_ego[:3, :3] = rig["rotation"]
        to_ego[:3, 3] = rig["translation"]
        cams[name] = dict(
            data_path=f"odyssey_probe/{name}/0.jpg" if name in cameras else "",
            sensor2lidar_rotation=np.array(rig["rotation"]), sensor2lidar_translation=np.array(rig["translation"]),
            cam_intrinsic=np.array(INTRINSICS), distortion=np.zeros(5), camera_to_ego=to_ego,
            sensor2ego_rotation=list(rig["quaternion"]), sensor2ego_translation=list(rig["translation"]),
            projection_semantics="training_undistorted", type=name)
    can_bus = np.zeros(18)
    can_bus[:2], can_bus[3], can_bus[10] = (x, y), 1.0, SPEED
    return dict(
        token="odyssey_probe-000", frame_idx=0, timestamp=0, log_name="odyssey_probe", log_token="odyssey_probe",
        scene_name="odyssey_probe", scene_token="odyssey_probe", map_location="", map_name="",
        roadblock_ids=[], route_roadblock_ids=[], vehicle_name="probe", can_bus=can_bus, lidar_path=None,
        lidar2ego_translation=np.zeros(3), lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0],
        ego2global_translation=np.array([x, y, 0.0]), ego2global_rotation=np.array([1.0, 0.0, 0.0, 0.0]),
        ego_dynamic_state=[SPEED, 0.0, 0.0, 0.0], traffic_lights=[], driving_command=np.array(command),
        command=int(np.argmax(command)), driving_command_source="probe", route_sidecar_token="odyssey_probe",
        cams=cams, ego2global=ego2global, lidar2ego=np.eye(4), lidar2global=ego2global, sweeps=[],
        sample_prev=None, sample_next=None)


def describe(value):
    if hasattr(value, "shape"):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[{len(value)}]"
    return type(value).__name__


def seed_all(seed=0):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def probe_worker(config, buffer):
    """Worker factory: a PlannerWorker that runs the probe at start-up (`metadata["probe"]`).

    The planner module (OpenCV, PIL, the model's imports) loads only here, in the planner
    interpreter, so `check` itself can import this module for `write_route`.
    """
    from .planner import PlannerWorker

    worker = type("ProbeWorker", (ProbeMixin, PlannerWorker), {})(config, buffer)
    if config.get("probe_dir"):
        worker.metadata["probe"] = worker.probe(config["probe_dir"])
    return worker


class ProbeMixin:
    """The probe steps, mixed into PlannerWorker by `probe_worker`."""

    def routed(self):
        """True when the planner reads the route through a route builder the probe can swap.

        The generic adapter and adapters that feed the route themselves (ReCogDrive) keep it in
        `_route_builder`, a StaticRouteCenterline over the run's route file.
        """
        return getattr(self.planner, "_route_builder", None) is not None

    def step(self, name, command, route_file, sensor_root):
        planner = self.planner
        if self.routed():
            from odyssey_bridge.route_sidecar import StaticRouteCenterline

            planner._route_builder = StaticRouteCenterline(
                getattr(planner._route_builder, "_config", None), sidecar_dir=route_file, strict=True)
        observed = {}
        planner.model_observer = lambda features, route, out: observed.update(
            {key: describe(value) for key, value in features.items()})
        from .planner import encode_jpeg

        blobs = {cam: encode_jpeg(synthetic_image(cam)) for cam in self.profile.cameras}
        for cam, data in blobs.items():            # adapters that open camera files by path
            path = os.path.join(sensor_root, f"odyssey_probe/{cam}/0.jpg")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as out:
                out.write(data)
        payload = b"".join(blobs.values())
        message = dict(step=0, episode=f"probe-{name}", frame=frame(command, blobs),
                       images=[(cam, len(data)) for cam, data in blobs.items()], payload_size=len(payload))
        seed_all()
        buffer, self.buffer = self.buffer, payload
        try:
            reply = self.handle(message)
        finally:
            self.buffer = buffer
        if reply.get("route_exhausted"):
            raise RuntimeError("the synthetic route was reported exhausted at the first step")
        return reply, observed

    def probe(self, probe_dir):
        result = dict(ok=False)
        sensor_root = self.config["sensor_root"]
        observer = self.planner.model_observer
        straight = write_route(os.path.join(probe_dir, "route_straight.npz"), turn=False)
        try:
            start = time.perf_counter()
            reply, observed = self.step("base", STRAIGHT, straight, sensor_root)
            result.update(ok=True, seconds=round(time.perf_counter() - start, 2),
                          planner_ms=round(reply["planner_ms"], 1), features=observed,
                          plan=np.asarray(reply["native"]).round(3).tolist(), wire_shape=list(reply["plan"].shape))
            base = np.asarray(reply["native"])
            if bool(self.planner.uses_driving_command):
                other, _ = self.step("command", LEFT, straight, sensor_root)
                result["command_changes_plan"] = not np.array_equal(base, np.asarray(other["native"]))
            if self.routed():
                left = write_route(os.path.join(probe_dir, "route_left.npz"), turn=True)
                other, _ = self.step("route", STRAIGHT, left, sensor_root)
                result["route_changes_plan"] = not np.array_equal(base, np.asarray(other["native"]))
        except Exception as error:
            result.update(ok=False, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        finally:
            self.planner.model_observer = observer
        return result
