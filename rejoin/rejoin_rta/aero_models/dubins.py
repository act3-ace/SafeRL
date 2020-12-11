import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math

class DubinsAircraft:

    def __init__(self, x=0, y=0, theta=0, velocity=100):

        self.vel_min = 10
        self.vel_max = 100

        self.reset(x, y, theta, velocity)

    def reset(self, x=0, y=0, theta=0, velocity=100):

        vel = velocity
        vel_dot = 0
        theta_dot = 0

        self.state = np.array([x, y, theta, vel, theta_dot, vel_dot], dtype=np.float64)

    def step(self, timestep, control=np.array([0, 0], dtype=np.float64)):
       # Extract current state data
        x, y, theta, vel, theta_dot, vel_dot = self.state

        # unpack control vector
        thrust_control = control[0]
        theta_control = control[1]

        # Euler Integrator
        # Integrate to calculate theta
        theta = theta + theta_control * timestep
        vel = vel + thrust_control * timestep

        # clip velocity to bounds
        vel = min(self.vel_max, max(self.vel_min, vel))

        # Calculate x-y velocities
        x_dot = vel * math.cos(theta)
        y_dot = vel * math.sin(theta)

        # Integrate to calculate x and y
        x = x + x_dot * timestep
        y = y + y_dot * timestep

        # save acceleration state
        vel_dot = thrust_control
        theta_dot = theta_control

        # x:0
        # y:1
        # theta:2
        # velocity: 3
        # theta_dot: 4
        # vel_dot: 5
        self.state = np.array([x, y, theta, vel, theta_dot, vel_dot], dtype=np.float64)

    def _generate_info(self):
        info = {
            'x': self.x,
            'y': self.y,
            'theta': self.theta,
            'vel': self.vel,
            'theta_dot': self.theta_dot,
            'vel_dot': self.vel_dot
        }

        return info

    @property
    def x(self):
        return self.state[0]

    @property
    def y(self):
        return self.state[1]

    @property
    def theta(self):
        return self.state[2]

    @property
    def vel(self):
        return self.state[3]

    @property
    def theta_dot(self):
        return self.state[4]

    @property
    def vel_dot(self):
        return self.state[5]
    
    @property
    def position(self) -> np.ndarray:
        return self.state[0:2]

    @property
    def orientation(self) -> np.ndarray:
        return self.theta

    @property
    def velocity_rect(self) -> np.ndarray:
        return self.vel * np.array([ math.cos(self.theta), math.sin(self.theta) ], dtype=np.float64)

    @property
    def velocity_polar(self) -> np.ndarray:
        return np.array([self.vel, self.theta], dtype=np.float64)

    @property
    def acceleration_rect(self) -> np.ndarray:
        return self.vel_dot * np.array([ math.cos(self.theta), math.sin(self.theta) ], dtype=np.float64)

    @property
    def acceleration_polar(self) -> np.ndarray:
        return np.array([self.vel_dot, self.theta], dtype=np.float64)

    def get_obs(self):
        return self.state

class DubinsAgent(DubinsAircraft):
    def __init__(self, action_type='Discrete', action_magnitude=[5, 1.5], **kwargs):
        self.action_type = action_type
        self.action_magnitude = action_magnitude # degrees/sec
        self.action_magnitude_radians = self.action_magnitude[1] * math.pi / 180       

        self.setup_action_space()     

        super(DubinsAgent, self).__init__(**kwargs)

    def setup_action_space(self):
        if self.action_type == 'Discrete':
            self.action_space = spaces.MultiDiscrete([5, 5]) # 5 discrete actions

            # create discrete action map
            self.discrete_action_space = [
                [
                    -2 * self.action_magnitude[0],
                    -1 * self.action_magnitude[0],
                    0 * self.action_magnitude[0],
                    1 * self.action_magnitude[0],
                    2 * self.action_magnitude[0]
                ],
                [
                    -2 * self.action_magnitude_radians,
                    -1 * self.action_magnitude_radians,
                    0,
                    1 * self.action_magnitude_radians,
                    2 * self.action_magnitude_radians
                ]
            ]

        else:
            # define continuous action space bounds
            self.cont_action_min = -2*self.action_magnitude
            self.cont_action_max = 2*self.action_magnitude

            self.action_space = spaces.Box(np.array([self.cont_action_min]), np.array([self.cont_action_max]), dtype=np.float64)

    def preprocess_action(self, action):
        if self.action_type == 'Discrete': # Discrete action space (Default)
            assert self.action_space.contains(action), "Invalid action"

            # map discrete action
            action_processed = np.zeros(len(self.discrete_action_space))
            action_processed[0] = self.discrete_action_space[0][int(action[0])]
            action_processed[1] = self.discrete_action_space[1][int(action[1])]

        else: # Continuous action space
            action = np.clip(action, self.cont_action_min, self.cont_action_max)
            action[0] = action[0] * math.pi / 180

        return action_processed


    def step(self, timestep, action=np.array([2,2])):

        action = self.preprocess_action(action)

        vel_dot = action[0]
        theta_dot = action[1]

        # pack control vector
        control = np.array([vel_dot, theta_dot], dtype=np.float64)

        super(DubinsAgent, self).step(timestep, control=control)