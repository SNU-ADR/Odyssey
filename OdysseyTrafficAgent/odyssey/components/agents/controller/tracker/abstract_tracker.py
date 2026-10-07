# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import abc


class AbstractTracker(abc.ABC):
    def __init__(self, agent):
        super(AbstractTracker, self).__init__()

        self.agent = agent

    @abc.abstractmethod
    def track_trajectory(self):
        pass
