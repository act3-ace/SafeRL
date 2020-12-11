import math
import numpy as np

import gym
from gym.spaces import Discrete, Box


from ..aero_models.dubins import DubinsAircraft, DubinsAgent
from ..utils.util import draw_from_rand_bounds_dict
from ..utils.geometry import RelativeCircle2D, distance2d

class DubinsRejoin(gym.Env):

    def __init__(self, config):
        # save config
        self.reward_config = config["reward_config"]
        self.rejoin_config = config['rejoin_config']

        if 'verbose' in config:
            self.verbose = config['verbose']
        else:
            self.verbose = False

        self.obs_integration = DubinsObservationIntegration(self.rejoin_config['obs'])
        self.reward_integration = DubinsRewardIntegration(self.reward_config)

        self._setup_env_objs()
        self._setup_action_space()
        self._setup_obs_space()

        self.reward_time_decay = -.01
        self.death_radius = 100

        self.reset()

    def _setup_env_objs(self):
        wingman = DubinsAgent()
        lead = DubinsAircraft()

        if self.rejoin_config['rejoin_region']['type'] == 'circle':
            r_offset = self.rejoin_config['rejoin_region']['range']
            radius = self.rejoin_config['rejoin_region']['radius']
            aspect_angle = np.radians(self.rejoin_config['rejoin_region']['aspect_angle'])
            rejoin_region = RelativeCircle2D(lead, radius=radius, track_orientation=True, r_offset=r_offset, aspect_angle=aspect_angle)
        else:
            raise ValueError('Invalid rejoin region type {} not supported'.format(self.rejoin_config['rejoin_region']['type']))

        self.env_objs = {
            'wingman': wingman,
            'lead': lead,
            'rejoin_region': rejoin_region,
        }

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
        self.status_dict = {}

        self.rejoin_time = 0
        self.reward_integration.reset(self.env_objs)
        
        obs = self._generate_obs()

        if self.verbose:
            print("env reset with params {}".format(self._generate_info()))

        return obs

    def step(self, action):
        timestep = 1

        self.env_objs['wingman'].step(timestep, action)
        self.env_objs['lead'].step(timestep)
        self.env_objs['rejoin_region'].step()

        # check rejoin condition
        in_rejoin = self.rejoin_cond()
        if in_rejoin:
            self.rejoin_time += timestep
        else:
            self.rejoin_time = 0

        # check success/failure conditions
        lead_distance =  distance2d(self.env_objs['wingman'], self.env_objs['lead'])
        
        if lead_distance < self.death_radius:
            failure = 'failure_crash'
        elif self.time_elapsed > 1000:
            failure = 'failure_timeout'
        elif lead_distance >= 40000:
            failure = 'failure_distance'
        else:
            failure = False


        if self.rejoin_time > 20:
            success = True
        else:
            success = False

        self.status_dict = {
            'success': success,
            'failure': failure,
            'in_rejoin': in_rejoin
        }

        reward = self._generate_reward(timestep)
        obs = self._generate_obs()
        info = self._generate_info()

        self.time_elapsed += timestep

        # determine if done
        if success or failure:
            done = True
        else:
            done = False

        return  obs, reward, done, info

    def _setup_obs_space(self):

        self.observation_space = self.obs_integration.observation_space

    def _setup_action_space(self):
        self.action_space = self.env_objs['wingman'].action_space
        
    def _generate_obs(self):
        obs = self.obs_integration.get_obs(self.env_objs)
        return obs

    def _generate_reward(self, timestep):
        reward = self.reward_integration.get_reward(self.env_objs, timestep, self.status_dict)
        return reward

    def _generate_info(self):

        info = {
            'wingman': self.env_objs['wingman']._generate_info(),
            'lead': self.env_objs['lead']._generate_info(),
            'rejoin_region': self.env_objs['rejoin_region']._generate_info(),
            'rejoin_time': self.rejoin_time,            
            'failure': self.status_dict['failure'],
            'success': self.status_dict['success'],
        }

        return info

    def rejoin_cond(self):
        wingman_coords = (self.env_objs['wingman'].x, self.env_objs['wingman'].y)
        return self.env_objs['rejoin_region'].contains(wingman_coords)


class DubinsObservationIntegration():
    def __init__(self, config):
        self.config = config

        if self.config['mode'] == 'rect':
            self.observation_space = Box(low=-1, high=1, shape=(8,))
            self.obs_norm_const = np.array([10000, 10000, 10000, 10000, 100, 100, 100, 100], dtype=np.float64)

        elif self.config['mode'] == 'polar':
            self.observation_space = Box(low=-1, high=1, shape=(12,))
            self.obs_norm_const = np.array([10000, 1, 1, 10000, 1, 1, 100, 1, 1, 100, 1, 1], dtype=np.float64)

    def get_obs(self, env_objs):
        wingman_lead_r_x = env_objs['lead'].x - env_objs['wingman'].x
        wingman_lead_r_y = env_objs['lead'].y - env_objs['wingman'].y

        wingman_rejoin_r_x = env_objs['rejoin_region'].x - env_objs['wingman'].x
        wingman_rejoin_r_y = env_objs['rejoin_region'].y - env_objs['wingman'].y

        if self.config['mode'] == 'rect':

            vel_rect = env_objs['wingman'].velocity_rect
            vel_x = vel_rect[0]
            vel_y = vel_rect[1]

            lead_vel_rect = env_objs['lead'].velocity_rect
            lead_vel_x = lead_vel_rect[0]
            lead_vel_y = lead_vel_rect[1]

            if self.config['reference'] == 'global':
                obs = np.array([wingman_lead_r_x, wingman_lead_r_y, wingman_rejoin_r_x, wingman_rejoin_r_y, vel_x, vel_y, lead_vel_x, lead_vel_y], dtype=np.float64)
            else:
                raise ValueError('Invalid obs referece {} for obs mode rect'.format(self.config['reference']))

        elif self.config['mode'] == 'polar':

            def rect2polar(vec_x, vec_y):
                mag = math.sqrt(vec_x**2 + vec_y**2)
                theta = math.atan2(vec_y,vec_x)
                return mag, theta

            def polar2obs(mag, theta):
                return [mag, math.cos(theta), math.sin(theta)]

            wingman_lead_r_mag, wingman_lead_r_theta = rect2polar(wingman_lead_r_x,wingman_lead_r_y)
            wingman_rejoin_r_mag, wingman_rejoin_r_theta = rect2polar(wingman_rejoin_r_x, wingman_rejoin_r_y)

            vel_polar = env_objs['wingman'].velocity_polar
            vel_mag = vel_polar[0]
            vel_theta = vel_polar[1]

            lead_vel_polar = env_objs['lead'].velocity_polar
            lead_vel_mag = lead_vel_polar[0]
            lead_vel_theta = lead_vel_polar[1]

            if self.config['reference'] == 'wingman':
                wingman_lead_r_theta -= vel_theta
                wingman_rejoin_r_theta -= vel_theta
                lead_vel_theta -= vel_theta
                vel_theta = 0
            else:
                raise ValueError('Invalid obs referece {} for obs mode polar'.format(self.config['reference']))

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

class DubinsRewardIntegration():
    def __init__(self, config):
        self.config = config

    def reset(self, env_objs):
        self.prev_distance = distance2d(env_objs['wingman'], env_objs['rejoin_region'])

        self.total_reward_rejoin = 0
        self.in_rejoin_prev = False
        self.rejoin_first_time_applied = False

    def get_reward(self, env_objs, timestep, status_dict):
        cur_distance = distance2d(env_objs['wingman'], env_objs['rejoin_region'])
        dist_change = cur_distance - self.prev_distance

        reward = 0

        reward += self.config['time_decay']

        self.prev_distance = cur_distance

        in_rejoin = status_dict['in_rejoin']

        if in_rejoin:
            reward_rejoin = self.config['rejoin_timestep'] * timestep
            reward += reward_rejoin
            self.total_reward_rejoin += reward_rejoin

            if not self.rejoin_first_time_applied:
                reward += self.config['rejoin_first_time']
                self.rejoin_first_time_applied = True
        else:
            reward_dist = dist_change*self.config['dist_change']
            reward += reward_dist

            if self.in_rejoin_prev:
                reward += -1*1*self.total_reward_rejoin
                self.total_reward_rejoin = 0

        self.in_rejoin_prev = in_rejoin

        if status_dict['failure']:
            reward += self.config[status_dict['failure']]
        elif status_dict['success']:
            reward += self.config['success']

        return reward