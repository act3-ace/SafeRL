import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math

class DubinsAircraft:

    def __init__(self, x=0, y=0, theta=0, velocity=100):

        self.reset(x, y, theta, velocity)

    def reset(self, x, y, theta, velocity):

        # Calculate velocities
        x_dot = velocity * math.cos(theta)
        y_dot = velocity * math.sin(theta)

        theta_dot = 0

        self.state = np.array([x, y, theta, x_dot, y_dot, theta_dot])

    def dynamic_step(self, timestep, control=np.array([0, 0])):
       # Extract current state data
        x, y, theta, x_dot, y_dot, theta_dot = self.state

        # unpack control vector
        vel_dot = control[0]
        theta_dot = control[1]

        # compute velocity magnitude
        velocity = math.sqrt(x_dot**2 + y_dot**2)

        # Euler Integrator
        # Integrate to calculate theta
        theta = theta + theta_dot * timestep
        velocity = velocity + vel_dot * timestep

        # Calculate velocities
        
        x_dot = velocity * math.cos(theta)
        y_dot = velocity * math.sin(theta)

        # Integrate to calculate x and y
        x = x + x_dot * timestep
        y = y + y_dot * timestep

        self.state = np.array([x, y, theta, x_dot, y_dot, theta_dot])

    def get_obs(self):

        return self.state

class DubinsAgent(DubinsAircraft):
    def __init__(self, action_type='Discrete', action_magnitude=1.5, **kwargs):
        self.action_type = action_type
        self.action_magnitude = action_magnitude # degrees/sec
        self.action_magnitude_radians = self.action_magnitude * math.pi / 180       

        self.setup_action_space()     

        super(DubinsAgent, self).__init__(**kwargs)

    def setup_action_space(self):
        if self.action_type == 'Discrete':
            self.action_space = spaces.Discrete(5) # 5 discrete actions

            # create discrete action map
            self.discrete_action_space = [
                -2 * self.action_magnitude_radians,
                -1 * self.action_magnitude_radians,
                0,
                1 * self.action_magnitude_radians,
                2 * self.action_magnitude_radians]

        else:
            # define continuous action space bounds
            self.cont_action_min = -2*self.action_magnitude
            self.cont_action_max = 2*self.action_magnitude

            self.action_space = spaces.Box(np.array([self.cont_action_min]), np.array([self.cont_action_max]), dtype=np.float64)

    def preprocess_action(self, action):
        if self.action_type == 'Discrete': # Discrete action space (Default)
            assert self.action_space.contains(action[0]), "Invalid action"

            # map discrete action
            action[0] = self.discrete_action_space[action[0]]

        else: # Continuous action space
            action = np.clip(action, self.cont_action_min, self.cont_action_max)
            action[0] = action[0] * math.pi / 180

        return action


    def step(self, timestep, action=np.array([0])):

        action = self.preprocess_action(action)

        vel_dot = 0
        theta_dot = action[0]

        # pack control vector
        control = np.array([vel_dot, theta_dot])

        self.dynamic_step(timestep, control=control)

        

