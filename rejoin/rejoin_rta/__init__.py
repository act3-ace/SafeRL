import abc
from scipy.spatial.transform import Rotation

class BaseEnvObj(abc.ABC):


    @property
    @abc.abstractmethod
    def x(self):
        ...

    @property
    @abc.abstractmethod
    def y(self):
        ...

    @property
    @abc.abstractmethod
    def position2d(self):
        ...

    @property
    @abc.abstractmethod
    def orientation(self) -> Rotation:
        ...

    @property
    def position(self):
        return self.position2d()

class BaseEnvObj3d(BaseEnvObj):

    @property
    @abc.abstractmethod
    def z(self):
        ...

    @property
    @abc.abstractmethod
    def position3d(self):
        ...

    @property
    def position(self):
        return self.position3d()