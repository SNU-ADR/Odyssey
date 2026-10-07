# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from odyssey.components.agents.vehicle_model.pacifica_vehicle import get_pacifica_parameters

def build_vehicle(object_id, config):
    if object_id == 'ego':
        vehicle_type = config.get('ego_vehicle_type', 'pacifica')
    elif object_id == 'controlled': # id is not ready yet.
        vehicle_type = config.get('other_vehicle_type', 'pacifica')
    else:
        vehicle_type = config.get('other_vehicle_type', 'pacifica')
        vehicle_type = None

    if vehicle_type == 'pacifica':
        return get_pacifica_parameters()
    else:
        return None