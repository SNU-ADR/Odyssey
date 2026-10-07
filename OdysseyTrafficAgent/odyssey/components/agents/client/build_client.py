# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from odyssey.components.agents.client.planner_client import PlannerClient
from odyssey.components.agents.client.base_client import BaseClient

def build_client(object_id: str, config):
    """
    Return the client class for target object.
    """
    if object_id == 'ego':
        client = config.get('ego_client')
    else:
        return BaseClient

    if client == 'planner_client':
        return PlannerClient
    elif client == 'base_client':
        return BaseClient
    else:
        raise NotImplementedError
