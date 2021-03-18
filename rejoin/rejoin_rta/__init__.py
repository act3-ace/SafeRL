import abc
from scipy.spatial.transform import Rotation
import numpy as np

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
    def z(self):
        ...

    @property
    @abc.abstractmethod
    def position(self):
        ...

    @property
    @abc.abstractmethod
    def orientation(self) -> Rotation:
        ...

    @property
    @abc.abstractmethod
    def velocity(self):
        ...