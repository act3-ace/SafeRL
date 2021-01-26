import gym
from gym import spaces
from gym.utils import seeding
import math
import numpy as np
from scipy import integrate

from rejoin_rta.aero_models import ContinuousActuator, PassThroughController, AgentController

def test_diff(t,x,a):
    return a

class CWHSpacecraft:
    def __init__(self, config=None, x=0, y=0, z=0, x_dot=0, y_dot=0, z_dot=0, precision=64):
        
        self.dependent_objs = []

        self.config = config

        if precision == 32:
            self.precision_dtype = np.float32
        elif precision == 64:
            self.precision_dtype = np.float64
        else:
            self.precision_dtype = np.float64

        self.dynamics = CWHDynamics()
        self.include_actuator_info = False

        if self.config is None or 'controller' not in self.config:
            controller_config = {
                'type': 'pass'
            }
        else:
            controller_config = self.config['controller']

        if controller_config['type'] == 'pass':
            controller = PassThroughController(config=controller_config)
        elif controller_config['type'] == 'agent':
            controller = AgentController(self.dynamics, config=controller_config)
            self.action_space = controller.action_space
            self.include_actuator_info = True

        self.controller = controller

        self.reset(x=x, y=y, z=z, x_dot=x_dot, y_dot=y_dot, z_dot=z_dot)

    def reset(self, x=0, y=0, z=0, x_dot=0, y_dot=0, z_dot=0):
        
        self.state = np.array([x, y, z, x_dot, y_dot, z_dot], dtype=self.precision_dtype)
        # self.controller.reset()

        for obj in self.dependent_objs:
            obj.reset()

    def step(self, step_size, action=None):

        control = self.controller.generate_control(self.state, action)

        self.state = self.dynamics.step(step_size, self.state, control)

    def register_dependent_obj(self, obj):
        self.dependent_objs.append(obj)

    def _generate_info(self):
        info = {
            'state':{
                'x': self.x,
                'y': self.y,
                'z': self.z,
                'x_dot': self.x_dot,
                'y_dot': self.y_dot,
                'z_dot': self.z_dot
            }
        }

        if self.include_actuator_info:
            info['actutators'] = self.dynamics.get_actuator_info()

        return info

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

class CWHDynamics:
    def __init__(self, precision_dtype = np.float64):
        self.m = 12 # kg
        self.n = 0.001027 # rads/s

        self.integration_method = 'rk45'
        self.precision_dtype = precision_dtype

        self.control_cur = None

        self.A, self.B = self.get_dynamics_matrices(self.m, self.n)

        self.actuators = self.define_actuators()
        self.default_control = np.array([0,0,0], dtype=np.float64)

    def step(self, step_size, state, control=None):

        if control is None:
            control = self.default_control

        # save the control action for the current timestep
        self.control_cur = np.copy(control)

        if self.integration_method == "rk45":
            sol = integrate.solve_ivp(self.dynamics_dx, (0,step_size), state, args=(control,))

            state = sol.y[:,-1] # save last timestep of integration solution 
        elif self.integration_method == 'euler': # euler
            state_dot = self.dynamics_dx(0, state, control)
            state = state + step_size * state_dot
        else:
            raise ValueError("invalid integration method '{}'".format(self.integration_method))

        return state

    def dynamics_dx(self, t, state, control):

        state_dot = np.matmul(self.A, state) + np.matmul(self.B, control)

        return state_dot

    def get_dynamics_matrices(self, m, n):
        A = np.array([
            [0,      0, 0,     1,     0,  0],
            [0,      0, 0,     0,     1,  0],
            [0,      0, 0,     0,     0,  1],
            [3*n**2, 0, 0,     0,    2*n, 0],
            [0,      0, 0,     -2*n, 0,   0],
            [0,      0, -n**2, 0,    0,   0],
        ], dtype=self.precision_dtype)

        B = np.array([
            [0,   0,   0   ],
            [0,   0,   0   ],
            [0,   0,   0   ],
            [1/m, 0,   0   ],
            [0,   1/m, 0   ],
            [0,   0,   1/m ],
        ], dtype=self.precision_dtype)

        return A, B

    def define_actuators(self):
        actuators = [
            ContinuousActuator(
                'thrust_x',
                [-100, 100]
            ),
            ContinuousActuator(
                'thrust_y',
                [-100, 100]
            ),
            ContinuousActuator(
                'thrust_z',
                [-100, 100]
            ),
        ]

        return actuators

    def get_actuator_info(self):
        if self.control_cur is None:
            info = None
        else:
            info = {}

            for i, actuator in enumerate(self.actuators):
                info[actuator.name] = self.control_cur[i]

        return info