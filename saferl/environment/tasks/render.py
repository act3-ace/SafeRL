import abc


class BaseRenderer(abc.ABC):
    """
    Class to define the BaseRender class template.
    """

    def __init__(self):
        self.viewer = None

    @abc.abstractmethod
    def renderSim(self, state, mode='human'):
        raise NotImplementedError

    def close(self):  # if a viewer exists, close and kill it
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def reset(self):
        self.close()
