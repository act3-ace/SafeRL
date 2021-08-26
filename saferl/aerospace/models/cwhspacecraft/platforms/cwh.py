import numpy as np
from scipy.spatial.transform import Rotation
import copy

from saferl.environment.models.platforms import BasePlatform, BasePlatformStateVectorized, ContinuousActuator, \
    BaseActuatorSet, BaseLinearODESolverDynamics


class BaseCWHSpacecraft(BasePlatform):

    def generate_info(self):
        info = {
            'state': self.state.vector,
            'x_dot': self.x_dot,
            'y_dot': self.y_dot,
        }

        info_parent = super().generate_info()
        info_ret = {**info_parent, **info}

        return info_ret

    @property
    def x_dot(self):
        return self.state.x_dot

    @property
    def y_dot(self):
        return self.state.y_dot


class CWHSpacecraft2d(BaseCWHSpacecraft):

    def __init__(self, name, controller=None):

        dynamics = CWH2dDynamics()
        actuator_set = CWH2dActuatorSet()
        state = CWH2dState()

        super().__init__(name, dynamics, actuator_set, state, controller)


class CWHSpacecraft3d(BaseCWHSpacecraft):

    def __init__(self, name, controller=None):
        dynamics = CWH3dDynamics()
        actuator_set = CWH3dActuatorSet()
        state = CWH3dState()

        super().__init__(name, dynamics, actuator_set, state, controller)

    def generate_info(self):
        info = {
            'z_dot': self.z_dot
        }

        info_parent = super().generate_info()
        info_ret = {**info_parent, **info}

        return info_ret

    @property
    def z_dot(self):
        return self.state.z_dot


class CWH2dState(BasePlatformStateVectorized):

    def build_vector(self, x=0, y=0, x_dot=0, y_dot=0, **kwargs):
        return np.array([x, y, x_dot, y_dot], dtype=np.float64)

    @property
    def x(self):
        return self._vector[0]

    @property
    def y(self):
        return self._vector[1]

    @property
    def z(self):
        return 0

    @property
    def x_dot(self):
        return self._vector[2]

    @property
    def y_dot(self):
        return self._vector[3]

    @property
    def position(self):
        position = np.zeros((3,))
        position[0:2] = self._vector[0:2]
        return position

    @property
    def orientation(self):
        # always return a no rotation quaternion as points do not have an orientation
        return Rotation.from_quat([0, 0, 0, 1])

    @property
    def velocity(self):
        vel = np.array([self.x_dot, self.y_dot, 0], dtype=np.float64)
        return vel


class CWH3dState(BasePlatformStateVectorized):

    def build_vector(self, x=0, y=0, z=0, x_dot=0, y_dot=0, z_dot=0, **kwargs):
        return np.array([x, y, z, x_dot, y_dot, z_dot], dtype=np.float64)

    @property
    def x(self):
        return self._vector[0]

    @property
    def y(self):
        return self._vector[1]

    @property
    def z(self):
        return self._vector[2]

    @property
    def x_dot(self):
        return self._vector[3]

    @property
    def y_dot(self):
        return self._vector[4]

    @property
    def z_dot(self):
        return self._vector[5]

    @property
    def position(self):
        return copy.deepcopy(self._vector[0:3])

    @property
    def orientation(self):
        # always return a no rotation quaternion as points do not have an orientation
        return Rotation.from_quat([0, 0, 0, 1])

    @property
    def velocity(self):
        return copy.deepcopy(self._vector[3:6])


class CWH2dActuatorSet(BaseActuatorSet):

    def __init__(self):
        actuators = [
            ContinuousActuator(
                'thrust_x',
                [-100, 100],
                0
            ),
            ContinuousActuator(
                'thrust_y',
                [-100, 100],
                0
            ),
        ]

        super().__init__(actuators)


class CWH3dActuatorSet(BaseActuatorSet):

    def __init__(self):
        actuators = [
            ContinuousActuator(
                'thrust_x',
                [-100, 100],
                0
            ),
            ContinuousActuator(
                'thrust_y',
                [-100, 100],
                0
            ),
            ContinuousActuator(
                'thrust_z',
                [-100, 100],
                0
            ),
        ]

        super().__init__(actuators)


class CWH2dDynamics(BaseLinearODESolverDynamics):
    def __init__(self, m=12, n=0.001027, integration_method='Euler'):
        self.m = m  # kg
        self.n = n  # rads/s

        super().__init__(integration_method=integration_method)

    def gen_dynamics_matrices(self):
        m = self.m
        n = self.n

        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [3 * n ** 2, 0, 0, 2 * n],
            [0, 0, -2 * n, 0],
        ], dtype=np.float64)

        B = np.array([
            [0, 0],
            [0, 0],
            [1 / m, 0],
            [0, 1 / m],
        ], dtype=np.float64)

        return A, B


class CWH3dDynamics(BaseLinearODESolverDynamics):
    def __init__(self, m=12, n=0.001027, integration_method='Euler'):
        self.m = m  # kg
        self.n = n  # rads/s

        super().__init__(integration_method=integration_method)

    def gen_dynamics_matrices(self):
        m = self.m
        n = self.n

        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3 * n ** 2, 0, 0, 0, 2 * n, 0],
            [0, 0, 0, -2 * n, 0, 0],
            [0, 0, -n ** 2, 0, 0, 0],
        ], dtype=np.float64)

        B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1 / m, 0, 0],
            [0, 1 / m, 0],
            [0, 0, 1 / m],
        ], dtype=np.float64)

        return A, B
