import math
import numpy as np
from scipy.spatial.transform import Rotation
import copy

from saferl.environment.models import BasePlatform, BasePlatformStateVectorized, ContinuousActuator, \
    BaseActuatorSet, BaseLinearODESolverDynamics

class CWHSpacecraftOriented2d(BasePlatform):

    def __init__(self, config=None, controller=None, **kwargs):
        self.mass = 12 # kg
        self.moment = 0.056 # kg*m^2
        self.react_wheel_moment = 4.1e-5 #kg*m^2
        self.n = 0.001027 #rad/s


        dynamics = CWHOriented2dDynamics(self)
        actuator_set = CWHOriented2dActuatorSet()

        state = CWHOriented2dState()

        super().__init__(dynamics, actuator_set, controller, state, config=config, **kwargs)

    def _generate_info(self):
        info = {
            'state': self.state.vector,
            'x': self.x,
            'y': self.y,
            'x_dot': self.x_dot,
            'y_dot': self.y_dot,
        }

        return info

    @property
    def x_dot(self):
        return self.state.x_dot

    @property
    def y_dot(self):
        return self.state.y_dot

class CWHOriented2dState(BasePlatformStateVectorized):

    def build_vector(self, x=0, y=0, theta=0, x_dot=0, y_dot=0, theta_dot=0, react_wheel_ang_vel=0, react_wheel_ang_acc=0, **kwargs):
        return np.array([x, y, theta, x_dot, y_dot, theta_dot, react_wheel_ang_vel, react_wheel_ang_acc], dtype=np.float64)

    @property
    def vector_shape(self):
        return (8,)

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
    def react_wheel_ang_vel(self):
        return self._vector[6]

    @property
    def react_wheel_ang_acc(self):
        return self._vector[7]

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

    def __init__(self):
        actuators = [
            ContinuousActuator(
                'thrust',
                [-100, 100],
                0
            ),
            ContinuousActuator(
                'reaction_wheel',
                [-100, 100],
                0
            ),
        ]

        super().__init__(actuators)

class CWHOriented2dDynamics(BaseLinearODESolverDynamics):
    def __init__( self, platform, integration_method='RK45'):
        self.platform = platform

        super().__init__(integration_method=integration_method)

    def dx(self, t, state_vec, control):
        state_cur = CWHOriented2dState(vector=state_vec, vector_deep_copy=False)
        
        pos_vel_state_vec = np.array( [state_cur.x, state_cur.y, state_cur.x_dot, state_cur.y_dot], dtype=np.float64 )

        thrust_vector = control[0] * np.array( [ math.cos(state_cur.theta), math.sin(state_cur.theta) ] )
        pos_vel_derivative = np.matmul( self.A, pos_vel_state_vec ) + np.matmul( self.B, thrust_vector )

        react_wheel_ang_acc = control[1]
        theta_dot_dot = -1 * self.platform.react_wheel_moment * react_wheel_ang_acc / self.platform.moment

        state_derivative = CWHOriented2dState(
            x = pos_vel_derivative[0],
            y = pos_vel_derivative[1],
            theta = state_cur.theta_dot,
            x_dot = pos_vel_derivative[2],
            y_dot = pos_vel_derivative[3],
            theta_dot = theta_dot_dot,
            react_wheel_ang_vel = state_cur.react_wheel_ang_acc,
            react_wheel_ang_acc = react_wheel_ang_acc,
        )

        return state_derivative.vector


    def gen_dynamics_matrices(self):
        m = self.platform.mass
        n = self.platform.n

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