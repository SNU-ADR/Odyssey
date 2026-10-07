from types import SimpleNamespace

import numpy as np
from nuplan.common.maps.maps_datatypes import TrafficLightStatusType

from odyssey.components.agents.policy.pdm_planner.utils.odyssey_to_pdm_utils import (
    OdysseyToNuPlanConverter,
)


def test_scene_conversion_reads_dimensions_from_current_step():
    """A track with zero-filled leading frames must not become a zero-sized box."""
    converter = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    converter.base_timestamp = 0
    converter.initial_ego_center = np.zeros(2)
    converter.scene = {
        "cadence": SimpleNamespace(sim_dt=0.1),
        "sample_rate": 2,
        "token": "scene",
        "object_track": {
            "late_vehicle": {
                "type": "VEHICLE",
                "metadata": {},
                "state": {
                    "position": np.array(
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 20.0, 1.0]]
                    ),
                    "heading": np.array([0.0, 0.0, 0.5]),
                    "velocity": np.array(
                        [[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]]
                    ),
                    "length": np.array([[0.0], [0.0], [4.8]]),
                    "width": np.array([[0.0], [0.0], [2.1]]),
                    "height": np.array([[0.0], [0.0], [1.7]]),
                },
            }
        },
    }

    tracks = converter.convert_to_detections_tracks_from_scene(2)
    vehicle = tracks.tracked_objects.tracked_objects[0]

    assert vehicle.box.length == 4.8
    assert vehicle.box.width == 2.1
    assert vehicle.box.height == 1.7


def test_scene_conversion_excludes_sdc_even_when_stored_as_vehicle():
    """Odyssey's VEHICLE-typed ego must not become a tracked traffic object."""
    converter = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    converter.base_timestamp = 0
    converter.initial_ego_center = np.zeros(2)
    state = {
        "position": np.array([[1.0, 2.0, 0.0]]),
        "heading": np.array([0.0]),
        "velocity": np.array([[0.0, 0.0]]),
        "length": np.array([[4.8]]),
        "width": np.array([[2.0]]),
        "height": np.array([[1.7]]),
    }
    converter.scene = {
        "cadence": SimpleNamespace(sim_dt=0.1),
        "sample_rate": 2,
        "token": "scene",
        "sdc_id": "ego",
        "object_track": {
            "ego": {"type": "VEHICLE", "metadata": {}, "state": state},
            "traffic": {"type": "VEHICLE", "metadata": {}, "state": state},
        },
    }

    tracks = converter.convert_to_detections_tracks_from_scene(0)
    tokens = [obj.track_token for obj in tracks.tracked_objects.tracked_objects]

    assert tokens == ["traffic"]


def test_scene_conversion_holds_final_frame_after_log_boundary():
    """The terminal bookkeeping step must not index one row beyond the source log."""
    converter = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    converter.base_timestamp = 0
    converter.initial_ego_center = np.zeros(2)
    converter.scene = {
        "cadence": SimpleNamespace(sim_dt=0.1),
        "sample_rate": 2,
        "token": "scene",
        "object_track": {
            "traffic": {
                "type": "VEHICLE",
                "metadata": {},
                "state": {
                    "position": np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]),
                    "heading": np.array([0.1, 0.2]),
                    "velocity": np.array([[1.0, 0.0], [2.0, 0.0]]),
                    "length": np.array([[4.0], [4.5]]),
                    "width": np.array([[1.8], [2.0]]),
                    "height": np.array([[1.5], [1.6]]),
                },
            }
        },
    }

    vehicle = converter.convert_to_detections_tracks_from_scene(2).tracked_objects.tracked_objects[0]

    assert np.allclose(vehicle.center.array, [3.0, 4.0])
    assert vehicle.center.heading == 0.2
    assert vehicle.box.length == 4.5


def test_traffic_light_conversion_marks_post_log_phase_unknown():
    """A long closed-loop rollout must not repeat a terminal RED indefinitely."""
    converter = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    converter.base_timestamp = 0
    converter.scene = {
        "cadence": SimpleNamespace(sim_dt=0.1),
        "dynamic_map_states": {
            "light": {
                "type": "TRAFFIC_LIGHT",
                "traffic_light_lane": "connector",
                "state": {
                    "traffic_light_state": [
                        "TRAFFIC_LIGHT_GREEN",
                        "TRAFFIC_LIGHT_RED",
                    ]
                },
            }
        },
    }

    assert converter.convert_to_traffic_lights(1)[0].status == TrafficLightStatusType.RED
    assert converter.convert_to_traffic_lights(2)[0].status == TrafficLightStatusType.UNKNOWN


def test_traffic_light_conversion_uses_the_shared_sector_source_row():
    converter = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    converter.base_timestamp = 0
    converter.scene = {
        "cadence": SimpleNamespace(sim_dt=0.1),
        "dynamic_map_states": {
            "light": {
                "type": "TRAFFIC_LIGHT",
                "traffic_light_lane": "connector",
                "state": {
                    "traffic_light_state": [
                        "TRAFFIC_LIGHT_GREEN",
                        "TRAFFIC_LIGHT_RED",
                    ]
                },
            }
        },
    }
    rows = {5: -1, 6: 0, 7: 1}
    converter.engine = SimpleNamespace(
        agent_manager=SimpleNamespace(
            traffic_light_source_row=lambda connector, step: rows[step]
        ),
        managers={},
    )

    assert converter.convert_to_traffic_lights(5)[0].status == TrafficLightStatusType.UNKNOWN
    assert converter.convert_to_traffic_lights(6)[0].status == TrafficLightStatusType.GREEN
    assert converter.convert_to_traffic_lights(7)[0].status == TrafficLightStatusType.RED


def test_scoring_observation_uses_executed_agent_pose_instead_of_log_gt():
    """Log replay and reactive IDM must both be scored at their current simulated pose."""
    converter = OdysseyToNuPlanConverter.__new__(OdysseyToNuPlanConverter)
    converter.base_timestamp = 0
    converter.initial_ego_center = np.array([1000.0, 2000.0])
    converter.scene = {
        "cadence": SimpleNamespace(sim_dt=0.1),
        "token": "scene",
        "sdc_id": "ego",
        "object_track": {
            "traffic": {
                "type": "VEHICLE",
                "metadata": {},
                # Deliberately unrelated to the executed closed-loop state below.
                "state": {
                    "position": np.array([[9000.0, 9000.0, 0.0]]),
                    "valid": np.array([True]),
                },
            }
        },
    }
    simulated_agent = SimpleNamespace(
        id="traffic",
        current_position=np.array([3.0, 4.0]),
        current_heading=0.25,
        current_velocity=np.array([5.0, 6.0]),
        _length=4.8,
        _width=2.0,
        _height=1.7,
    )
    converter.engine = SimpleNamespace(
        agent_manager=SimpleNamespace(all_agents={"traffic": simulated_agent}),
        managers={},
    )

    tracked = converter.convert_to_detections_tracks_from_agent_input(0)
    vehicle = tracked.tracked_objects.tracked_objects[0]

    assert np.allclose(vehicle.center.array, [1003.0, 2004.0])
    assert vehicle.center.heading == 0.25
    assert np.allclose(vehicle.velocity.array, [5.0, 6.0])
