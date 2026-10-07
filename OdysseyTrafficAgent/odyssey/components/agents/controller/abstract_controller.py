# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import abc


class AbstractController(abc.ABC):
    """
    Interface for generic ego controllers.
    """

    def __init__(self, agent):
        self.agent = agent

    @abc.abstractmethod
    def step(self):
        """
        Update ego's state from current iteration to next iteration.
        """
        pass

    @property
    def name(self):
        return self.__class__.__name__
