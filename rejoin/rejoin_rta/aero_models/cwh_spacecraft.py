import gym
from gym import spaces
from gym.utils import seeding
import math
import numpy as np
from scipy.spatial.transform import Rotation

from rejoin_rta.aero_models import ContinuousActuator, PassThroughController, AgentController, BaseActuatorSet, BaseLinearODESolverDynamics, BasePlatform

def test_diff(t,x,a):
    return a

class CWHSpacecraft(BasePlatform):


    def __init__(self, config=None, **kwargs):
        dynamics = CWH3dDynamics()
        actuator_set = CWH3dActuatorSet()

        if config is None or 'controller' not in config:
            controller_config = {
                'type': 'pass'
            }
        else:
            controller_config = config['controller']

        if controller_config['type'] == 'pass':
            controller = PassThroughController(config=controller_config)
        elif controller_config['type'] == 'agent':
            controller = AgentController(actuator_set, config=controller_config)
            self.action_space = controller.action_space

        super().__init__(dynamics, actuator_set, controller, init_dict=kwargs)

    def init_state(self, x=0, y=0, z=0, x_dot=0, y_dot=0, z_dot=0):
        
        return np.array([x, y, z, x_dot, y_dot, z_dot], dtype=np.float64)

    def _generate_info(self):
        info = {
            'state': self.state,
            'x': self.x,
            'y': self.y,
            'z': self.z,
            'x_dot': self.x_dot,
            'y_dot': self.y_dot,
            'z_dot': self.z_dot
        }

        return info

    # def estimate_trajectory(self, time_window=20, num_points=None):
    #     return self.dynamics.estimate_trajectory(self.state, time_window=time_window, num_points=num_points)

    @property
    def x(self):
        return self.state[0]

    @property
    def y(self):
        return self.state[1]

    @property
    def z(self):
        return self.state[2]

    @property
    def x_dot(self):
        return self.state[3]

    @property
    def y_dot(self):
        return self.state[4]

    @property
    def z_dot(self):
        return self.state[5]

    @property
    def position(self):
        return self.position3d

    @property
    def position2d(self):
        return self.state[0:2]

    @property
    def position3d(self):
        return self.state[0:3]

    @property
    def state2d(self):
        return self.state[ [0, 1, 3 ,4] ]

    @property
    def orientation(self):
        # always return a no rotation quaternion as points do not have an orientation
        return Rotation.from_quat([0, 0, 0, 1])


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

class CWH3dDynamics(BaseLinearODESolverDynamics):
    def __init__(self, integration_method = 'RK45'):
        self.m = 12 # kg
        self.n = 0.001027 # rads/s

        super().__init__(integration_method=integration_method)

    def gen_dynamics_matrices(self):

        m = self.m
        n = self.n

        A = np.array([
            [0,      0, 0,     1,     0,  0],
            [0,      0, 0,     0,     1,  0],
            [0,      0, 0,     0,     0,  1],
            [3*n**2, 0, 0,     0,    2*n, 0],
            [0,      0, 0,     -2*n, 0,   0],
            [0,      0, -n**2, 0,    0,   0],
        ], dtype=np.float64)

        B = np.array([
            [0,   0,   0   ],
            [0,   0,   0   ],
            [0,   0,   0   ],
            [1/m, 0,   0   ],
            [0,   1/m, 0   ],
            [0,   0,   1/m ],
        ], dtype=np.float64)

        return A, B