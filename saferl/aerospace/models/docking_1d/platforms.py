import numpy as np
from scipy.spatial.transform import Rotation
import copy

from saferl.environment.models.platforms import BasePlatform, BasePlatformStateVectorized, ContinuousActuator, \
    BaseActuatorSet, BaseLinearODESolverDynamics


class BaseSpacecraft(BasePlatform):

    def generate_info(self):
        info = {
            'state': self.state.vector,
            'x_dot': self.x_dot,
        }

        info_parent = super().generate_info()
        info_ret = {**info_parent, **info}

        return info_ret

    @property
    def x_dot(self):
        return self.state.x_dot


class Spacecraft1D(BaseSpacecraft):

    def __init__(self, name, controller=None, integration_method='Euler'):

        dynamics = Dynamics1D(integration_method=integration_method)
        actuator_set = ActuatorSet1D()
        state = State1D()

        super().__init__(name, dynamics, actuator_set, state, controller)



class State1D(BasePlatformStateVectorized):

    def build_vector(self, x=0, x_dot=0, **kwargs):
        return np.array([x, x_dot], dtype=np.float64)

    @property
    def x(self):
        return self._vector[0]

    @property
    def x_dot(self):
        return self._vector[1]

    @property
    def position(self):
        position = np.zeros((3,))
        position[0] = self._vector[0]
        return position

    @property
    def orientation(self):
        # always return a no rotation quaternion as points do not have an orientation
        return Rotation.identity()

    @property
    def velocity(self):
        vel = np.array([self.x_dot, 0, 0], dtype=np.float64)    # array sizes fixed??*
        return vel

    @property
    def y(self):
        return 0

    @property
    def z(self):
        return 0


class ActuatorSet1D(BaseActuatorSet):

    def __init__(self):
        actuators = [
            ContinuousActuator(
                'thrust_x',
                [-1, 1],
                0
            ),
        ]

        super().__init__(actuators)



class Dynamics1D(BaseLinearODESolverDynamics):
    def __init__(self, m=12, integration_method='Euler'):
        self.m = m  # kg

        super().__init__(integration_method=integration_method)

    def gen_dynamics_matrices(self):
        m = self.m
        n = self.n

        A = np.array([
            [0, 1],
            [0, 0],
        ], dtype=np.float64)

        B = np.array([
            [0],
            [1 / self.m],
        ], dtype=np.float64)

        return A, B
