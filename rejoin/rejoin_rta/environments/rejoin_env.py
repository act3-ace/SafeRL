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

        self.death_radius = 25
        self.rejoin_max_radius = 150
        self.rejoin_min_radius = 50

        self.reward_config = config["reward_config"]

        self.reset()

    def step(self, action):
        timestep = 1
        
        self.env_objs['wingman'].step(timestep, action)
        self.env_objs['lead'].step(timestep)

        # check rejoin condition
        self.rejoin_failed = False
        self.in_rejoin = self.rejoin_cond()
        if self.in_rejoin:
            self.rejoin_time += timestep
        elif self.rejoin_time > 0:
            self.rejoin_time = 0
            self.rejoin_failed = True

        # check success/failure conditions
        distance =  np.linalg.norm(self.env_objs['wingman'].position - self.env_objs['lead'].position)
        if distance >= 10000:
            self.failure = 'failure_distance'
        if self.time_elapsed > 1000:
            self.failure = 'failure_timeout'
        if distance < self.death_radius:
            self.failure = 'failure_crash'

        if self.rejoin_time > 20:
            self.success = True

        obs = self._generate_obs()
        reward = self._generate_reward(timestep)
        info = self._generate_info()

        self.time_elapsed += timestep

        # determine if done
        done = False
        if self.success or self.failure:
            done = True

        return  obs, reward, done, info


    def reset(self):

        self.env_objs['wingman'].reset(x=0, y=1000, theta=0, velocity=100)
        self.env_objs['lead'].reset(x=1000, y=0, theta=0, velocity=50)

        self.time_elapsed = 0
        self.success = False
        self.failure = False
        
        self.rejoin_time = 0
        self.in_rejoin = False
        self.rejoin_failed = False
        self.total_reward_rejoin = 0
        self.rejoin_first_time_applied = False

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

        obs = np.clip(obs, -1, 1)

        return obs

    def _init_reward(self):
        self.prev_distance = np.linalg.norm(self.env_objs['wingman'].position - self.env_objs['lead'].position)

    def _generate_reward(self, timestep):
        cur_distance = np.linalg.norm(self.env_objs['wingman'].position - self.env_objs['lead'].position)
        dist_change = cur_distance - self.prev_distance

        reward = 0

        reward += self.reward_time_decay

        self.prev_distance = cur_distance

        if self.in_rejoin:
            reward_rejoin = self.reward_config['rejoin_timestep'] * timestep
            reward += reward_rejoin
            self.total_reward_rejoin += reward_rejoin

            if not self.rejoin_first_time_applied:
                reward += self.reward_config['rejoin_first_time']
                self.rejoin_first_time_applied = True
        else:
            reward_dist = dist_change*self.reward_config['dist_change']
            reward += reward_dist

            if self.rejoin_failed:
                reward += -1*1*self.total_reward_rejoin
                self.total_reward_rejoin = 0

        if self.failure:
            reward += self.reward_config[self.failure]
        elif self.success:
            reward += self.reward_config['success']

        return reward

    def _generate_info(self):

        info = {
            'wingman': self.env_objs['wingman']._generate_info(),
            'lead': self.env_objs['lead']._generate_info(),
            'rejoin_time':self.rejoin_time,
        }

        return info

    def rejoin_cond(self):
        distance =  np.linalg.norm(self.env_objs['wingman'].position - self.env_objs['lead'].position)

        if distance > self.rejoin_min_radius and distance < self.rejoin_max_radius:
            return True
        else:
            return False