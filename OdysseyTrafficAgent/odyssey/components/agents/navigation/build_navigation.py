# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from odyssey.components.agents.navigation.trajectory_navigation import TrajectoryNavigation


def build_navigation(object_id: str, config):
    """
    Return the navigation class for target object.
    """

    if object_id == 'ego':
        navigation = config.get('ego_navigation', 'trajectory_navigation')
    else:
        navigation = config.get('agent_navigation', 'trajectory_navigation')

    if navigation == 'trajectory_navigation':
        return TrajectoryNavigation
    else:
        raise NotImplementedError(f'The assigned navigation {navigation} is not'
                                  f'implemented.')
