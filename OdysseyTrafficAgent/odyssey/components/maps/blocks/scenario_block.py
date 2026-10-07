# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
"""
Definition of block objects, used for managing multiple roads within a block.
"""

import logging
import math
from abc import ABC

import cv2
import numpy as np

from odyssey.utils.type import OdysseyObjectType
from odyssey.scenario.scenarios.scenario_description import ScenarioDescription as SD

from odyssey.components.maps.blocks.base_block import BaseBlock
from odyssey.components.maps.road_networks.edge_road_network import EdgeRoadNetwork
from odyssey.components.maps.lanes.scenario_lane import ScenarioLane


class ScenarioBlock(BaseBlock):
    """
    Base class for RoadBlock.
    """

    object_ID = "RoadBlock_"

    def __init__(self, block_index,
                 global_network,
                 random_seed,
                 map_index,
                 map_data,):
        super(ScenarioBlock, self).__init__(block_index, global_network, random_seed)

        self.map_index = map_index
        self.map_data = map_data

        self._sample_topology()
        self.global_network.add(self.block_network, no_intersect=True)

    def _sample_topology(self) -> bool:
        """
        Sample a new topology to fill self.block_network
        """
        # for each map element.
        for object_id, data in self.map_data.items():
            if OdysseyObjectType.is_lane(data.get('type', False)):
                if len(data[SD.POLYLINE]) <= 1:
                    continue
                lane = ScenarioLane(object_id, self.map_data)
                self.block_network.add_lane(lane)

            elif OdysseyObjectType.is_sidewalk(data["type"]):
                self.sidewalks[object_id] = {
                    SD.TYPE: OdysseyObjectType.BOUNDARY_SIDEWALK,
                    SD.POLYGON: np.asarray(data[SD.POLYGON])[..., :2]
                }

            elif OdysseyObjectType.is_crosswalk(data["type"]):
                self.crosswalks[object_id] = {
                    SD.TYPE: OdysseyObjectType.CROSSWALK,
                    SD.POLYGON: np.asarray(data[SD.POLYGON])[..., :2]
                }
            else:
                pass
        return True

    @property
    def block_network_type(self):
        return EdgeRoadNetwork



