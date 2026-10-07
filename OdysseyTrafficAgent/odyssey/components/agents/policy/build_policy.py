# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from odyssey.components.agents.policy.trajectory_policy import TrajectoryPolicy
from odyssey.components.agents.policy.pdm_policy import PDMPolicy
from odyssey.components.agents.policy.env_input_policy import EnvInputPolicy
from odyssey.components.agents.policy.nuplan_idm_policy import NuPlanIDMPolicy

def build_policy(object_id: str, config):
    """
    Return the navigation class for target object.
    """

    if object_id == 'ego':
        policy = config.get('ego_policy')
    else:
        policy = config.get('agent_policy', 'trajectory_policy')

    if policy == 'trajectory_policy':
        return TrajectoryPolicy
    elif policy == 'idm_policy':
        raise ValueError(
            "Odyssey's legacy per-agent idm_policy was removed. "
            "Use agent_policy=nuplan_idm_policy for reactive background traffic. "
            "nuplan_idm_policy is not an ego planner."
        )
    elif policy == 'pdm_policy':
        return PDMPolicy
    elif policy == 'nuplan_idm_policy':
        if object_id == 'ego':
            raise ValueError(
                "nuplan_idm_policy manages reactive background traffic only; "
                "choose an ego planner such as pdm_policy for ego_policy."
            )
        # nuPlan's IDM: sees traffic lights and cross traffic, and indexes the lead
        # search with an STRtree instead of walking every lane. See the module docstring.
        return NuPlanIDMPolicy
    elif policy == 'env_input_policy':
        return EnvInputPolicy
    else:
        raise NotImplementedError(f'The assigned policy {policy} is not'
                                  f'implemented.')
