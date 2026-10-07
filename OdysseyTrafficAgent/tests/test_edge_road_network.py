import numpy as np

from odyssey.components.maps.lanes.center_lane import CenterLane
from odyssey.components.maps.road_networks.edge_road_network import EdgeRoadNetwork


def _lane(index, points):
    lane = CenterLane(np.asarray(points, dtype=np.float64), width=3.5)
    lane.index = index
    lane.entry_lanes = []
    lane.exit_lanes = []
    lane.left_lanes = []
    lane.right_lanes = []
    return lane


def _scalar_results(network, position):
    return sorted(
        (
            lane_info.lane.distance(position),
            lane_index,
            lane_info.lane,
        )
        for lane_index, lane_info in network.graph.items()
    )


def test_vectorized_lane_distances_match_scalar_order_and_values():
    network = EdgeRoadNetwork()
    network.add_lane(_lane("straight", [[-4, 0], [0, 0], [7, 0]]))
    network.add_lane(_lane("corner", [[0, -5], [0, 0], [4, 4], [8, 4]]))
    network.add_lane(_lane("parallel", [[-4, 3], [8, 3]]))

    random = np.random.default_rng(7)
    for position in random.uniform(-10.0, 10.0, size=(100, 2)):
        expected = _scalar_results(network, position)
        actual = network.get_closest_lane_index(position, return_all=True)
        assert [row[1] for row in actual] == [row[1] for row in expected]
        np.testing.assert_allclose(
            [row[0] for row in actual],
            [row[0] for row in expected],
            rtol=0.0,
            atol=1e-12,
        )
        assert network.get_closest_lane_index(position)[1] == expected[0][1]


def test_lane_localization_cache_is_invalidated_when_lane_is_added():
    network = EdgeRoadNetwork()
    network.add_lane(_lane("far", [[100, 0], [110, 0]]))
    assert network.get_closest_lane_index([0, 0])[1] == "far"

    network.add_lane(_lane("near", [[0, 0], [10, 0]]))
    assert network.get_closest_lane_index([0, 0])[1] == "near"
