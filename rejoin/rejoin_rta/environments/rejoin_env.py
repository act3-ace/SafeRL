import math
import numpy as np

import gym
from gym.spaces import Discrete, Box


from ..aero_models.dubins import DubinsAircraft, DubinsAgent
from ..utils.util import draw_from_rand_bounds_dict

class DubinsRejoin(gym.Env):

    def __init__(self, config):

        self.env_objs = {
            'wingman':DubinsAgent(),
            'lead': DubinsAircraft()
        }

        self.action_space = self.env_objs['wingman'].action_space

        self.reward_time_decay = -.01

        self.death_radius = 100

        self.reward_config = config["reward_config"]
        self.rejoin_config = config['rejoin_config']

        if 'verbose' in config:
            self.verbose = config['verbose']
        else:
            self.verbose = False

        self._obs_space_init()
        self.reset()

    def reset(self):

        init_dict = self.rejoin_config['init']

        successful_init = False
        while not successful_init:
            init_dict_draw = draw_from_rand_bounds_dict(init_dict)
            for obj_key, obj_init_dict in init_dict_draw.items():
                self.env_objs[obj_key].reset(**obj_init_dict)

            # TODO check if initialization is safe
            successful_init = True

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

        if self.verbose:
            print("env reset with params {}".format(self._generate_info()))

        return obs

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
        if distance >= 40000:
            self.failure = 'failure_distance'
        if self.time_elapsed > 1000:
            self.failure = 'failure_timeout'
        if distance < self.death_radius:
            self.failure = 'failure_crash'

        if self.rejoin_time > 20:
            self.success = True

        reward = self._generate_reward(timestep)
        obs = self._generate_obs()
        info = self._generate_info()

        self.time_elapsed += timestep

        # determine if done
        done = False
        if self.success or self.failure:
            done = True

        return  obs, reward, done, info

    def _obs_space_init(self):

        
        if self.rejoin_config['obs']['mode'] == 'rect':
            self.observation_space = Box(low=-1, high=1, shape=(8,))
            self.obs_norm_const = np.array([10000, 10000, 10000, 10000, 100, 100, 100, 100], dtype=np.float64)

        elif self.rejoin_config['obs']['mode'] == 'polar':
            self.observation_space = Box(low=-1, high=1, shape=(12,))
            self.obs_norm_const = np.array([10000, 1, 1, 10000, 1, 1, 100, 1, 1, 100, 1, 1], dtype=np.float64)
        

    def _generate_obs(self):
        wingman_lead_r_x = self.env_objs['lead'].x - self.env_objs['wingman'].x
        wingman_lead_r_y = self.env_objs['lead'].y - self.env_objs['wingman'].y

        wingman_rejoin_r_x = self.rejoin_point_pos[0] - self.env_objs['wingman'].x
        wingman_rejoin_r_y = self.rejoin_point_pos[1] - self.env_objs['wingman'].y

        if self.rejoin_config['obs']['mode'] == 'rect':

            vel_rect = self.env_objs['wingman'].velocity_rect
            vel_x = vel_rect[0]
            vel_y = vel_rect[1]

            lead_vel_rect = self.env_objs['lead'].velocity_rect
            lead_vel_x = lead_vel_rect[0]
            lead_vel_y = lead_vel_rect[1]

            if self.rejoin_config['obs']['reference'] == 'global':
                obs = np.array([wingman_lead_r_x, wingman_lead_r_y, wingman_rejoin_r_x, wingman_rejoin_r_y, vel_x, vel_y, lead_vel_x, lead_vel_y], dtype=np.float64)
            else:
                raise ValueError('Invalid obs referece {} for obs mode rect'.format(self.rejoin_config['obs']['reference']))

        elif self.rejoin_config['obs']['mode'] == 'polar':

            def rect2polar(vec_x, vec_y):
                mag = math.sqrt(vec_x**2 + vec_y**2)
                theta = math.atan2(vec_y,vec_x)
                return mag, theta

            def polar2obs(mag, theta):
                return [mag, math.cos(theta), math.sin(theta)]

            wingman_lead_r_mag, wingman_lead_r_theta = rect2polar(wingman_lead_r_x,wingman_lead_r_y)
            wingman_rejoin_r_mag, wingman_rejoin_r_theta = rect2polar(wingman_rejoin_r_x, wingman_rejoin_r_y)

            vel_polar = self.env_objs['wingman'].velocity_polar
            vel_mag = vel_polar[0]
            vel_theta = vel_polar[1]

            lead_vel_polar = self.env_objs['lead'].velocity_polar
            lead_vel_mag = lead_vel_polar[0]
            lead_vel_theta = lead_vel_polar[1]

            if self.rejoin_config['obs']['reference'] == 'wingman':
                wingman_lead_r_theta -= vel_theta
                wingman_rejoin_r_theta -= vel_theta
                lead_vel_theta -= vel_theta
                vel_theta = 0
            else:
                raise ValueError('Invalid obs referece {} for obs mode polar'.format(self.rejoin_config['obs']['reference']))

            obs =np.array(
                    polar2obs(wingman_lead_r_mag, wingman_lead_r_theta) +
                    polar2obs(wingman_rejoin_r_mag, wingman_rejoin_r_theta) +
                    polar2obs(vel_mag, vel_theta) +
                    polar2obs(lead_vel_mag, lead_vel_theta)
                )

        # normalize observation
        obs = np.divide(obs, self.obs_norm_const)

        obs = np.clip(obs, -1, 1)

        return obs

    def _init_reward(self):
        self.prev_distance = self.compute_rejoin_dist()

    def compute_rejoin_dist(self):
        # compute rejoin point location
        lead_pos = self.env_objs['lead'].position
        lead_orientation = self.env_objs['lead'].orientation
        rejoin_point_angle_rad = (self.rejoin_config['rejoin_point_aspect_angle'] - 180) * math.pi/180 + lead_orientation

        rejoin_point_range = self.rejoin_config['rejoin_point_range']
        rejoin_point_pos = rejoin_point_range*np.array([math.cos(rejoin_point_angle_rad), math.sin(rejoin_point_angle_rad) ], dtype=np.float64) + lead_pos

        self.rejoin_point_pos = rejoin_point_pos

        distance = np.linalg.norm(self.env_objs['wingman'].position - rejoin_point_pos)
        return distance

    def _generate_reward(self, timestep):
        cur_distance = self.compute_rejoin_dist()
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
            'rejoin_point_pos': self.rejoin_point_pos.tolist(),
            'failure': self.failure,
            'success': self.success,
        }

        return info

    def rejoin_cond(self):
        distance =  self.compute_rejoin_dist()

        if distance <= self.rejoin_config['rejoin_point_radius']:
            return True
        else:
            return False
