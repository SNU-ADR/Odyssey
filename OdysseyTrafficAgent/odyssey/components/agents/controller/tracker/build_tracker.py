# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
from odyssey.components.agents.controller.tracker.lqr_tracker import LQRTracker


def build_tracker(config):

    tracker = config.get('tracker', 'lqr')
    if tracker == 'lqr':
        return LQRTracker
    else:
        raise NotImplementedError(f'The assigned tracker {tracker} is not'
                                  f'implemented.')
