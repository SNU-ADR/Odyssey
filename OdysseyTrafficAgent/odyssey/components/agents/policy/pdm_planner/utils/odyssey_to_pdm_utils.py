# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint, StateVector2D
from nuplan.common.actor_state.ego_state import EgoState
from odyssey.components.agents.vehicle_model.pacifica_vehicle import get_pacifica_parameters
from odyssey.utils import math_utils
from odyssey.common.dataclasses import Trajectory
from odyssey.utils.cadence import PLANNER_POSE_DT
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType

from typing import List, Dict, Union
import numpy as np

TrackedObject = Union[Agent, StaticObject]

tracked_object_type_mapping = {
    "VEHICLE": TrackedObjectType.VEHICLE,
    "PEDESTRIAN": TrackedObjectType.PEDESTRIAN,
    "CYCLIST": TrackedObjectType.BICYCLE,
    "BICYCLE": TrackedObjectType.BICYCLE,  # alias -- some converters (e.g. digitaltwin/OmniRe) label this type directly
    "TRAFFIC_CONE": TrackedObjectType.TRAFFIC_CONE,
    "BARRIER": TrackedObjectType.BARRIER,
    "TRAFFIC_BARRIER": TrackedObjectType.BARRIER,
    "CZONE_SIGN": TrackedObjectType.CZONE_SIGN,
    "GENERIC_OBJECT": TrackedObjectType.GENERIC_OBJECT,
    "TRAFFIC_OBJECT": TrackedObjectType.GENERIC_OBJECT,
    "EGO": TrackedObjectType.EGO
}

# Utils
def _sim_dt(scene):
    """Outer step length (s) of this scene, set once by ScenarioManager via resolve_cadence.

    Do not rebuild it as scene["sample_rate"] * 0.05: the formula is right (sample_rate counts
    0.05 s periods), but deriving the same number in several places eventually diverges.
    If it is missing, fail instead of guessing.
    """
    c = scene.get("cadence") if hasattr(scene, "get") else None
    if c is None:
        raise RuntimeError("scene['cadence'] is missing; sim_dt is not re-derived here.")
    return float(c.sim_dt)


def denormalize_from_ego_center(vector, ego_center):
    "Denormalize position-related vectors from the ego_center of the first frame."
    vector = np.array(vector)
    vector += np.asarray(ego_center)
    return vector

def normalize_to_ego_center(vector, ego_center=(0, 0)):
    "Normalize position-related vectors to the ego_center of the first frame."
    vector = np.array(vector)
    vector -= np.asarray(ego_center)
    return vector


class OdysseyToNuPlanConverter:
    def __init__(self, scene, agent, engine):
        self.scene = scene
        self.agent = agent
        self.engine = engine
        self.base_timestamp = scene.get("base_timestamp", 0.0)  # Default 0 if not present
        self.initial_ego_center = - self.scene["metadata"]["old_origin_in_current_coordinate"] # take the opposite number
        # ``convert_to_detections_tracks_from_scene`` is called at every reactive-IDM tick.
        # The source scene is immutable, and walking every track in Python only to discover
        # that most have a zero-filled pose at this frame is slow. Build a compact validity table lazily and retain the source dictionary's order.  This
        # is only a candidate index: the nuPlan Agent/box below is still constructed from the
        # exact same per-frame source values, so observation cadence and geometry are unchanged.
        self._scene_track_ids = None
        self._scene_track_valid = None
    def _scene_track_items_at(self, current_step: int):
        """Return source tracks present at ``current_step`` in original dictionary order.

        Rows shorter than the longest source track repeat their final validity bit.  That
        deliberately matches the converter's existing ``min(current_step, len(position)-1)``
        behavior, including BaseEnv's final ungraded step past the recorded window.
        """
        # Some lightweight callers/tests construct the converter with ``__new__`` and
        # provide only the data members needed by this conversion path.
        if getattr(self, "_scene_track_ids", None) is None:
            object_track = self.scene["object_track"]
            sdc_id = self.scene.get("sdc_id")
            ids = [
                object_id for object_id, track in object_track.items()
                if object_id != sdc_id
                and bool(track.get("metadata", {}).get("simulation_enabled", True))
            ]
            lengths = [len(object_track[object_id]["state"]["position"]) for object_id in ids]
            max_length = max(lengths, default=0)
            valid = np.zeros((len(ids), max_length), dtype=np.bool_)
            for row, (object_id, length) in enumerate(zip(ids, lengths)):
                if length == 0:
                    continue
                positions = np.asarray(object_track[object_id]["state"]["position"])
                row_valid = np.any(positions != 0.0, axis=1)
                valid[row, :length] = row_valid
                if length < max_length:
                    valid[row, length:] = row_valid[-1]
            self._scene_track_ids = tuple(ids)
            self._scene_track_valid = valid

        if self._scene_track_valid.shape[1] == 0:
            return ()
        active_rows = []
        for index, object_id in enumerate(self._scene_track_ids):
            source_row = self.object_source_row(object_id, current_step)
            if (source_row >= 0
                    and self._scene_track_valid[
                        index, min(source_row, self._scene_track_valid.shape[1] - 1)
                    ]):
                active_rows.append(index)
        object_track = self.scene["object_track"]
        return tuple(
            (self._scene_track_ids[row], object_track[self._scene_track_ids[row]])
            for row in active_rows
        )

    def object_source_row(self, object_id: str, simulation_step: int) -> int:
        """Resolve the source row actually executed for one open-loop object."""
        engine = getattr(self, 'engine', None)
        managers = getattr(engine, 'managers', {})
        manager = managers.get('agent_manager') if hasattr(managers, 'get') else None
        resolver = getattr(manager, 'object_source_row', None)
        if resolver is None:
            return int(simulation_step)
        return int(resolver(str(object_id), int(simulation_step)))

    def convert_to_current_ego_state(self, current_step: int) -> EgoState:
        """
        Convert Odyssey scene data to nuPlan's initial_ego_state format.

        Args:
            current_step: Current simulation step to extract the corresponding state.

        Returns:
            EgoState: nuPlan-compatible initial_ego_state object.
        """
        # Extract ego vehicle's data
        ego_id = self.scene["sdc_id"]
        ego_state = self.scene["object_track"][ego_id]["state"]
        

        # Center position and heading
        position = self.agent.rear_vehicle.current_position
        rear_position = denormalize_from_ego_center(position, self.initial_ego_center)  # Denormalize from ego_center
        heading = self.agent.current_heading   # First frame heading
        
        # Transfer to rear_axle
        rear_velocity = self.agent.current_velocity
        cos, sin = np.cos(heading), np.sin(heading)
        rear_velocity = np.array([
            cos * rear_velocity[0] + sin * rear_velocity[1],
            -sin * rear_velocity[0] + cos * rear_velocity[1]
        ])

        # Acceleration, rotated global->ego exactly like the velocity above. Do not zero it:
        # the PDM comfort sub-metrics for lon/lat acceleration and both jerk terms read
        # StateIndex.ACCELERATION_{X,Y}, so a zero here would make them pass unconditionally.
        rear_accel = np.asarray(self.agent.rear_vehicle.current_acceleration, dtype=float).reshape(-1)[:2]
        rear_accel = np.array([
            cos * rear_accel[0] + sin * rear_accel[1],
            -sin * rear_accel[0] + cos * rear_accel[1]
        ])

        # Timestamp
        timestamp = self.base_timestamp + current_step * _sim_dt(self.scene) * 1e6 
        # scene["sample_rate"] * 0.05 represents sim_dt

        # Construct StateSE2
        state_se2 = StateSE2(x=rear_position[0], y=rear_position[1], heading=heading)

        # Construct EgoState
        current_ego_state = EgoState.build_from_rear_axle(
            rear_axle_pose=state_se2,
            rear_axle_velocity_2d=StateVector2D(x=rear_velocity[0], y=rear_velocity[1]),
            rear_axle_acceleration_2d=StateVector2D(x=rear_accel[0], y=rear_accel[1]),
            tire_steering_angle=float(self.agent.current_tire_steering),
            vehicle_parameters=get_pacifica_parameters(),
            time_point=TimePoint(timestamp)
        )

        return current_ego_state

    def convert_to_detections_tracks_from_scene(self, current_step: int) -> DetectionsTracks:
        """
        Convert Odyssey object_track data to nuPlan DetectionsTracks format.

        Args:
            current_step: Current simulation step to extract the corresponding state.

        Returns:
            DetectionsTracks: A DetectionsTracks object for the current step.
        """
        tracked_objects = []  # List of TrackedObject
        # Timestamp
        timestamp = self.base_timestamp + current_step * _sim_dt(self.scene) * 1e6 
        for object_id, object_data in self._scene_track_items_at(current_step):
            # The active-track index already excludes the SDC.  nuPlan observations contain
            # tracked objects around the ego, never the ego itself; admitting Odyssey's
            # logged SDC would create a second ego after closed-loop divergence.
            # Extract and map object type
            object_type = object_data["type"]
            tracked_object_type = tracked_object_type_mapping.get(object_type)
            if tracked_object_type is None:
                raise ValueError(f"Unknown object type: {object_type}")
            # Extract state for the current step
            state = object_data["state"]
            # BaseEnv performs one final, ungraded step before publishing its truncation flag.
            # On an upsampled 160-frame cl80 scene the available rows are 0..795, so that final
            # call arrives as 796. Hold the last recorded observation, matching traffic-light
            # conversion below, rather than losing the whole batch to an IndexError after its
            # score has already been written.
            source_row = self.object_source_row(object_id, current_step)
            if source_row < 0:
                continue
            state_step = min(source_row, len(state["position"]) - 1)
            position = state["position"][state_step]  # [x, y, z]
            heading = state.get("heading", [0] * len(state["position"]))[state_step]  # Default 0 if not present
            velocity = state.get("velocity", np.zeros_like(state["position"]))[state_step]
            # Dimensions are stored per frame just like position and velocity.  Reading row 0
            # here made every actor which first appears later in the scene a zero-sized box,
            # because invalid leading frames are deliberately zero-filled.  In particular this
            # produced IDMAgents with width == 0, whose path.buffer(width / 2) is empty and trips
            # nuPlan's self-intersection invariant as soon as the actor is admitted.
            length = state.get("length", np.zeros((len(state["position"]), 1)))[state_step]
            width = state.get("width", np.zeros((len(state["position"]), 1)))[state_step]
            height = state.get("height", np.zeros((len(state["position"]), 1)))[state_step]

            position = denormalize_from_ego_center(position[:2], self.initial_ego_center)

            metadata = object_data["metadata"]

            # Create OrientedBox
            box = OrientedBox(
                center = StateSE2(x=position[0], y=position[1], heading=heading),
                length = float(length[0]),
                width = float(width[0]),
                height = float(height[0])
            )

            # Create TrackedObject based on type
            if object_type in ["VEHICLE", "PEDESTRIAN", "CYCLIST", "BICYCLE", "EGO"]:
                tracked_object = Agent(
                    tracked_object_type=tracked_object_type,
                    oriented_box=box,
                    velocity=StateVector2D(x=velocity[0], y=velocity[1]),
                    metadata=SceneObjectMetadata(
                        timestamp_us = timestamp,
                        token = self.scene["token"],
                        track_id = metadata.get("nuplan_id", object_id),
                        track_token = object_id
                    )
                )
            else:
                tracked_object = StaticObject(
                    tracked_object_type = tracked_object_type,
                    oriented_box=box,
                    metadata=SceneObjectMetadata(
                        timestamp_us = timestamp,
                        token = self.scene["token"],
                        track_id = metadata.get("nuplan_id", object_id),
                        track_token = object_id
                    )
                )

            # Append to the tracked objects list
            tracked_objects.append(tracked_object)

        # Wrap in TrackedObjects
        tracked_objects_container = TrackedObjects(tracked_objects=tracked_objects)

        # Wrap in DetectionsTracks
        detections_tracks = DetectionsTracks(tracked_objects=tracked_objects_container)


        return detections_tracks
    
    def _track_len(self):
        t = getattr(self, "_track_len_v", None)
        if t is None:
            any_tr = next(iter(self.scene["object_track"].values()), None)
            t = self._track_len_v = (
                len(np.asarray(any_tr["state"]["valid"]).ravel()) if any_tr else 0)
        return t

    def _rendered_tokens(self):
        """Set of actor tokens the renderer drew this frame; None if there is no renderer or it
        cannot tell.

        Returns actor tokens, not node names -- active_tokens are asset node keys
        (deform_object_<token> etc.), so the token is stripped out here.
        """
        rm = self.engine.managers.get("render_manager") if hasattr(self.engine, "managers") else None
        renderer = getattr(rm, "renderer", None)
        fn = getattr(renderer, "rendered_tokens", None)
        if fn is None:
            return None
        toks = fn()
        if toks is None:
            return None
        return {t.split("_object_")[-1] if "_object_" in t else t for t in toks}

    def convert_to_detections_tracks_from_agent_input(self, current_step) -> DetectionsTracks:
        """
        Convert real-time agent states from engine to nuPlan's DetectionsTracks format.
        This function differs from convert_to_detections_tracks_from_scene as it reads current states
        directly from running agents in the engine instead of pre-recorded scene data.

        Args:
            current_step (int): Current simulation step to calculate the timestamp

        Returns:
            DetectionsTracks: A container of all tracked objects (agents) in nuPlan format, including:
                - Dynamic objects (vehicles, pedestrians, cyclists)
                - Static objects (traffic cones, barriers)
                Each object contains:
                - Position, heading, and velocity
                - Object type and dimensions
                - Metadata (timestamp, track_id, etc.)
        """
        tracked_objects = []  # List of TrackedObject
        # Timestamp
        timestamp = self.base_timestamp + current_step * _sim_dt(self.scene) * 1e6
        agent_map = {agent.id: agent for agent in self.engine.agent_manager.all_agents.values()}

        # The scorer does not decide by itself whom to count. It scores every actor the simulator
        # currently holds, at the position it received. Log replay decides that set and those
        # positions in log mode, IDM in IDM mode -- both are decided outside the scorer.
        #
        # (The track's valid flag is not used as a filter. In log replay it happens to match
        #  "rendered actors", but IDM keeps driving cars whose log track has ended, so it would
        #  drop actually moving cars from scoring. Invisible actors are already filtered out by the scenario builder via the ckpt node set, so there is no
        #  reason to filter them again here.)
        #
        # The render list is only cross-checked (None = no renderer, or nothing drawn yet).
        # MetricManager runs before RenderManager, so the list is one frame late; mismatches
        # accumulating well beyond that one-beat lag at appearance signal "a scored car is not on
        # screen".
        visible = self._rendered_tokens()
        skipped_invisible = 0
        skipped_unset_pose = 0
        scored_ids, skipped_ids = [], []

        for object_id, object_data in self.scene["object_track"].items():
            if object_id == self.scene.get("sdc_id"):
                continue
            agent = agent_map.get(object_id)
            if agent is None:
                continue
            if visible is not None and object_id not in visible:
                skipped_invisible += 1
                skipped_ids.append(object_id)

            position = agent.current_position
            velocity = agent.current_velocity
            # The local origin means "no pose", not "located there". An actor whose source has
            # ended sometimes reports (0,0) for one frame just before disappearing, and
            # denormalize_from_ego_center maps that to initial_ego_center -- where the rollout
            # started. Early on the ego is still there, so an off-screen actor is recorded as
            # embedded in the ego and the scorer counts it as a contact.
            #
            # Log validity (valid) is not used as a filter, for the reason in the comment above.
            # What is checked here is not the track's valid interval but only **whether a pose
            # arrived this frame**. Cars IDM keeps driving, and parked cars that stay in place
            # after their source ends, both report their own coordinates and are scored as usual.
            #
            # Skip only when the velocity is also 0. Checking the origin alone could drop "an
            # actor that happens to be at the start point", whereas an actor without a pose has no
            # velocity either (its velocity is exactly 0).
            # Same signature as the rescoring side (drop_unset_pose_frames).
            if not np.any(position) and not np.any(velocity):
                skipped_unset_pose += 1
                continue

            scored_ids.append(object_id)

            object_type = object_data["type"]
            tracked_object_type = tracked_object_type_mapping.get(object_type)
            if tracked_object_type is None:
                raise ValueError(f"Unknown object type: {object_type}")
            heading = agent.current_heading
            length = agent._length
            width = agent._width
            height = agent._height

            position = denormalize_from_ego_center(position, self.initial_ego_center)

            # Create OrientedBox
            box = OrientedBox(
            center = StateSE2(x=position[0], y=position[1], heading=heading),
            length = float(length),
            width = float(width),
            height = float(height)
            )

            # Create TrackedObject based on type
            if object_type in ["VEHICLE", "PEDESTRIAN", "CYCLIST", "BICYCLE", "EGO"]:
                tracked_object = Agent(
                    tracked_object_type=tracked_object_type,
                    oriented_box=box,
                    velocity=StateVector2D(x=velocity[0], y=velocity[1]),
                    metadata=SceneObjectMetadata(
                        timestamp_us = timestamp,
                        token = self.scene["token"],
                        track_id = object_id,
                        track_token = object_id
                    )
                )
            else:
                tracked_object = StaticObject(
                    tracked_object_type = tracked_object_type,
                    oriented_box=box,
                    metadata=SceneObjectMetadata(
                        timestamp_us = timestamp,
                        token = self.scene["token"],
                        track_id = object_id,
                        track_token = object_id
                    )
                )

            # Append to the tracked objects list
            tracked_objects.append(tracked_object)

        # Wrap in TrackedObjects
        tracked_objects_container = TrackedObjects(tracked_objects=tracked_objects)

        # Wrap in DetectionsTracks
        detections_tracks = DetectionsTracks(tracked_objects=tracked_objects_container)

        # Accumulate how much was filtered. MetricManager summarises it in one line at the end
        # of the run, so a silent filter cannot go unnoticed.
        st = getattr(self, "visibility_stats", None)
        if st is None:
            st = self.visibility_stats = {"frames": 0, "frames_filtered": 0,
                                          "scored": 0,
                                          "skipped_invisible": 0,
                                          "skipped_unset_pose": 0,
                                          "ever_scored": set(), "ever_skipped": set()}
        st["frames"] += 1
        if visible is not None:
            st["frames_filtered"] += 1
        st["scored"] += len(tracked_objects)
        st["skipped_invisible"] += skipped_invisible
        st.setdefault("skipped_unset_pose", 0)
        st["skipped_unset_pose"] += skipped_unset_pose
        st["ever_scored"].update(scored_ids)
        st["ever_skipped"].update(skipped_ids)

        return detections_tracks
    
    def traffic_light_source_row(self, connector_id: str, current_step: int) -> int:
        """Resolve one environment signal to its scenario-PKL source row.

        AgentManager owns the replay clock because it releases the actor cohorts.  Keeping
        the lookup here gives ego planning, IDM and TLC one signal-state entry point.  Small
        standalone converter users and non-sector runs retain the identity clock.
        """
        engine = getattr(self, "engine", None)
        manager = getattr(engine, "agent_manager", None)
        if manager is None and engine is not None:
            managers = getattr(engine, "managers", {})
            manager = managers.get("agent_manager") if hasattr(managers, "get") else None
        resolver = getattr(manager, "traffic_light_source_row", None)
        if resolver is None:
            return int(current_step)
        return int(resolver(str(connector_id), int(current_step)))

    def convert_to_traffic_lights(self, current_step: int) -> DetectionsTracks:
        """
        Convert Odyssey traffic_light data to nuPlan DetectionsTracks format.

        Args:
            current_step: Current simulation step to extract the corresponding state.

        Returns:
            DetectionsTracks: A DetectionsTracks object for the current step.
        """
        traffic_lights = []
        # Timestamp
        timestamp = self.base_timestamp + current_step * _sim_dt(self.scene) * 1e6
        
        for _, map_object_data in self.scene["dynamic_map_states"].items():
            if map_object_data["type"] == "TRAFFIC_LIGHT":
                state = map_object_data["state"]
                traffic_light_state = state["traffic_light_state"]
                connector_id = str(map_object_data["traffic_light_lane"])
                source_row = self.traffic_light_source_row(connector_id, current_step)

                tracked_map_object_type_mapping = {
                "TRAFFIC_LIGHT_UNKNOWN": TrafficLightStatusType.UNKNOWN,
                "TRAFFIC_LIGHT_GREEN": TrafficLightStatusType.GREEN,
                "TRAFFIC_LIGHT_RED": TrafficLightStatusType.RED,
                }

                # A slow closed-loop ego may outlive the recorded signal timeline.  Repeating
                # the final phase is unsafe: a terminal RED becomes a synthetic 100-second red
                # light and can strand the whole route.  Pose replay may hold its final sample,
                # but a traffic-light phase is time-varying evidence; once that evidence is
                # exhausted it is UNKNOWN.  The IDM adapter then applies its existing
                # unprotected-intersection fallback, while TLC retains its separate ``held``
                # audit bit and therefore never mistakes this padding for a measured phase.
                if source_row < 0 or source_row >= len(traffic_light_state):
                    raw_status = "TRAFFIC_LIGHT_UNKNOWN"
                else:
                    raw_status = traffic_light_state[source_row]
                status = tracked_map_object_type_mapping.get(raw_status)
                if status is None:
                    raise ValueError(f"Unknown object type: {raw_status}")
                traffic_light = TrafficLightStatusData(
                    status = status,
                    lane_connector_id = connector_id,
                    timestamp = timestamp,
                )
                traffic_lights.append(traffic_light)
        
        return traffic_lights
    
    def convert_to_trajectory(self, pdm_path, wp_dt=PLANNER_POSE_DT):

        """
        Convert InterpolatedTrajectory to Trajectory.
        """
        pdm_trajectory = pdm_path._trajectory
        waypoints = []
        velocities = []
        headings = []
        for _, ego_state in enumerate(pdm_trajectory):
            waypoint = ego_state.waypoint
            x_world, y_world = waypoint.x, waypoint.y
            position = normalize_to_ego_center([x_world, y_world], self.initial_ego_center)
            waypoints.append(position)
            velocities.append([waypoint.velocity.x, waypoint.velocity.y])
            headings.append(waypoint.heading)
        
        waypoints = np.array(waypoints, dtype=np.float32)
        velocities = np.array(velocities, dtype=np.float32)
        headings = np.array(headings, dtype=np.float32)

        return Trajectory(
                waypoints=waypoints,
                velocities=velocities,
                headings=headings,
                angular_velocities=None,
                # The caller passes the planner's native grid.  PDM-Closed is 10 Hz even when
                # MetricManager grades saved rollouts on its separate 0.5 s score grid.
                wp_dt=float(wp_dt),
        )
