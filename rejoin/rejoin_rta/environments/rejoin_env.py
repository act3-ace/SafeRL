import math
import numpy as np

import gym
from gym.spaces import Discrete, Box


from ..aero_models.dubins import DubinsAircraft, DubinsAgent

class DubinsRejoin(gym.Env):

    def __init__(self, config):

        self.env_objs = {
            'wingman':DubinsAgent(x=0, y=1000, theta=0, velocity=100),
            'lead': DubinsAircraft(x=1000, y=0, theta=0, velocity=50)
        }

        self.action_space = self.env_objs['wingman'].action_space
        self.observation_space = Box(low=-1, high=1, shape=(4,))

        self.obs_norm_const = np.array([10000, 10000, 100, 100], dtype=np.float64)
        self.reward_time_decay = -.01

        self.time_elapsed = 0

        self._init_reward()

    def step(self, action):
        timestep = 1
        
        self.env_objs['wingman'].step(timestep, action)
        self.env_objs['lead'].step(timestep)

        obs = self._generate_obs()
        reward = self._generate_reward()
        info = self._generate_info()

        self.time_elapsed += timestep

        # determine if done
        done = False
        distance =  np.linalg.norm(self.env_objs['wingman'].position - self.env_objs['lead'].position)
        if distance >= 10000 or distance <= 5 or self.time_elapsed > 1000:
            done = True
            if distance <= 5:
                reward += 100

        return  obs, reward, done, info


    def reset(self):

        self.env_objs['wingman'].reset(x=0, y=1000, theta=0, velocity=100)
        self.env_objs['lead'].reset(x=1000, y=0, theta=0, velocity=50)

        self.time_elapsed = 0

        self._init_reward()
        obs = self._generate_obs()

        return obs

    def _generate_obs(self):
        r_x = self.env_objs['lead'].x - self.env_objs['wingman'].x
        r_y = self.env_objs['lead'].y - self.env_objs['wingman'].y

        vel_rect = self.env_objs['wingman'].velocity_rect
        vel_x = vel_rect[0]
        vel_y = vel_rect[1]

        obs = np.array([r_x, r_y, vel_x, vel_y], dtype=np.float64)

        # normalize observation
        obs = np.divide(obs, self.obs_norm_const)

        return obs

    def _init_reward(self):
        self.prev_distance = np.linalg.norm(self.env_objs['wingman'].position - self.env_objs['lead'].position)

    def _generate_reward(self):
        cur_distance = np.linalg.norm(self.env_objs['wingman'].position - self.env_objs['lead'].position)
        dist_change = cur_distance - self.prev_distance

        reward = -1*dist_change/100

        reward += self.reward_time_decay

        self.prev_distance = cur_distance

        return reward

    def _generate_info(self):

        info = {
            'wingman': self.env_objs['wingman']._generate_info(),
            'lead': self.env_objs['lead']._generate_info()
        }

        return info