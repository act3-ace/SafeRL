import math
import numpy as np
from scipy.spatial.transform import Rotation

from saferl.environment.models.platforms import BasePlatform, BasePlatformStateVectorized, ContinuousActuator, \
    BaseActuatorSet, BaseLinearODESolverDynamics
from saferl.aerospace.models.cwhspacecraft.platforms.cwh import CWH2dDynamics


class CWHSpacecraftOriented2d(BasePlatform):

    def __init__(self, name, controller=None, integration_method='RK45', **kwargs):
        self.mass = 12  # kg
        self.moment = 0.056  # kg*m^2
        self.react_wheel_moment = 4.1e-5  # kg*m^2
        self.react_wheel_ang_acc_limit = 181.3  # rad/s^2
        self.react_wheel_ang_vel_limit = 576  # rad/s
        self.n = 0.001027  # rad/s

        ang_acc_limit = min(np.deg2rad(1), self.react_wheel_moment * self.react_wheel_ang_acc_limit / self.moment)
        ang_vel_limit = min(np.deg2rad(2), self.react_wheel_moment * self.react_wheel_ang_vel_limit / self.moment)

        dynamics = CWHOriented2dDynamics(
            ang_vel_limit=ang_vel_limit, m=self.mass, n=self.n, integration_method=integration_method)
        actuator_set = CWHOriented2dActuatorSet(ang_acc_limit=ang_acc_limit)

        state = CWHOriented2dState()

        super().__init__(name, dynamics, actuator_set, state, controller)

    def generate_info(self):
        info = {
            'state': self.state.vector,
            'theta': self.theta,
            'x_dot': self.x_dot,
            'y_dot': self.y_dot,
            'theta_dot': self.theta_dot,
        }

        info_parent = super().generate_info()
        info_ret = {**info_parent, **info}

        return info_ret

    @property
    def theta(self):
        return self.state.theta

    @property
    def x_dot(self):
        return self.state.x_dot

    @property
    def y_dot(self):
        return self.state.y_dot

    @property
    def theta_dot(self):
        return self.state.theta_dot


class CWHOriented2dState(BasePlatformStateVectorized):

    def build_vector(self, x=0, y=0, theta=0, x_dot=0, y_dot=0, theta_dot=0, **kwargs):
        return np.array([x, y, theta, x_dot, y_dot, theta_dot], dtype=np.float64)

    @property
    def vector_shape(self):
        return (6,)

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
    def theta(self):
        return self._vector[2]

    @property
    def x_dot(self):
        return self._vector[3]

    @property
    def y_dot(self):
        return self._vector[4]

    @property
    def theta_dot(self):
        return self._vector[5]

    @property
    def position(self):
        position = np.zeros((3,))
        position[0:2] = self._vector[0:2]
        return position

    @property
    def orientation(self):
        return Rotation.from_euler('z', self.theta)

    @property
    def velocity(self):
        vel = np.array([self.x_dot, self.y_dot, 0], dtype=np.float64)
        return vel


class CWHOriented2dActuatorSet(BaseActuatorSet):

    def __init__(self, ang_acc_limit):
        assert isinstance(ang_acc_limit, float), "ang_acc_limit must be type float"
        assert ang_acc_limit > 0, "ang_acc_limit must be positive"

        actuators = [
            ContinuousActuator(
                'thrust',
                [-1, 1],
                0
            ),
            ContinuousActuator(
                'reaction_wheel',
                [-ang_acc_limit, ang_acc_limit],
                0
            ),
        ]

        super().__init__(actuators)


class CWHOriented2dDynamics(CWH2dDynamics):
    def __init__(self, ang_vel_limit, m=12, n=0.001027, integration_method='RK45'):
        self.ang_vel_limit = ang_vel_limit

        super().__init__(m=m, n=n, integration_method=integration_method)

    def dx(self, t, state_vec, control):
        state_cur = CWHOriented2dState(vector=state_vec, vector_deep_copy=False)

        pos_vel_state_vec = np.array([state_cur.x, state_cur.y, state_cur.x_dot, state_cur.y_dot], dtype=np.float64)

        thrust_vector = control[0] * np.array([math.cos(state_cur.theta), math.sin(state_cur.theta)])
        pos_vel_derivative = np.matmul(self.A, pos_vel_state_vec) + np.matmul(self.B, thrust_vector)

        theta_dot_dot = control[1]

        # check angular velocity limit
        if state_cur.theta_dot >= self.ang_vel_limit:
            theta_dot_dot = min(0, theta_dot_dot)
        elif state_cur.theta_dot <= self.ang_vel_limit:
            theta_dot_dot = max(0, theta_dot_dot)

        state_derivative = CWHOriented2dState(
            x=pos_vel_derivative[0],
            y=pos_vel_derivative[1],
            theta=state_cur.theta_dot,
            x_dot=pos_vel_derivative[2],
            y_dot=pos_vel_derivative[3],
            theta_dot=theta_dot_dot,
        )

        return state_derivative.vector
