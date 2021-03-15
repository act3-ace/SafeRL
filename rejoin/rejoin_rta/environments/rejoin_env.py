import math
import numpy as np
import random

import gym
from gym.spaces import Discrete, Box


from ..aero_models.dubins import DubinsAircraft, DubinsAgent
from ..utils.util import draw_from_rand_bounds_dict
from ..utils.geometry import RelativeCircle2D, distance2d

class DubinsRejoin(gym.Env):

    def __init__(self, config):
        # save config
        self.config = config

        if 'verbose' in config:
            self.verbose = config['verbose']
        else:
            self.verbose = False

        self.obs_integration = DubinsObservationIntegration(self.config['obs'])
        self.reward_integration = DubinsRewardIntegration(self.config["reward"])
        self.constraints_integration = DubinsConstraintIntegration(self.config['constraints'])

        self._setup_env_objs()
        self._setup_action_space()
        self._setup_obs_space()

        self.reset()

    def _setup_env_objs(self):
        wingman = DubinsAgent()
        lead = DubinsAircraft()

        if self.config['rejoin_region']['type'] == 'circle':
            r_offset = self.config['rejoin_region']['range']
            radius = self.config['rejoin_region']['radius']
            aspect_angle = np.radians(self.config['rejoin_region']['aspect_angle'])
            rejoin_region = RelativeCircle2D(lead, radius=radius, track_orientation=True, r_offset=r_offset, aspect_angle=aspect_angle)
        else:
            raise ValueError('Invalid rejoin region type {} not supported'.format(self.config['rejoin_region']['type']))

        self.env_objs = {
            'wingman': wingman,
            'lead': lead,
            'rejoin_region': rejoin_region,
        }

    def seed(self, seed=None):
        np.random.seed(seed)
        # note that python random should not be used (use numpy random instead)
        # Setting seed just to be safe in case it is accidentally used
        random.seed(seed)

        return [seed]

    def reset(self):
        init_dict = self.config['init']

        successful_init = False
        while not successful_init:
            init_dict_draw = draw_from_rand_bounds_dict(init_dict)
            for obj_key, obj_init_dict in init_dict_draw.items():
                self.env_objs[obj_key].reset(**obj_init_dict)

            # TODO check if initialization is safe
            successful_init = True

        self.timestep = 1       # sec

        self.status_dict = {}

        self.reward_integration.reset(self.env_objs)
        self.obs_integration.reset()
        self.constraints_integration.reset()
        
        obs = self._generate_obs()

        if self.verbose:
            print("env reset with params {}".format(self._generate_info()))

        return obs

    def step(self, action):
        self.env_objs['lead'].step(self.timestep)
        self.env_objs['wingman'].step(self.timestep, action)

        self.status_dict = self._generate_constraint_status()

        reward = self._generate_reward()
        obs = self._generate_obs()
        info = self._generate_info()

        # determine if done
        if self.status_dict['success'] or self.status_dict['failure']:
            done = True
        else:
            done = False

        return obs, reward, done, info

    def _setup_obs_space(self):
        self.observation_space = self.obs_integration.observation_space

    def _setup_action_space(self):
        self.action_space = self.env_objs['wingman'].action_space
        
    def _generate_obs(self):
        obs = self.obs_integration.gen_obs(self.env_objs)
        return obs

    def _generate_reward(self):
        reward = self.reward_integration.gen_reward(self.env_objs, self.timestep, self.status_dict)
        return reward

    def _generate_constraint_status(self):
        return self.constraints_integration.step(self.env_objs, self.timestep)

    def _generate_info(self):

        info = {
            'wingman': self.env_objs['wingman']._generate_info(),
            'lead': self.env_objs['lead']._generate_info(),
            'rejoin_region': self.env_objs['rejoin_region']._generate_info(),
            'failure': self.status_dict['failure'],
            'success': self.status_dict['success'],
            'status': self.status_dict,
            'reward': self.reward_integration._generate_info(),
            'timestep_size': self.timestep
        }

        return info

class DubinsObservationIntegration():
    def __init__(self, config):
        self.config = config

        if self.config['mode'] == 'rect':
            self.observation_space = Box(low=-1, high=1, shape=(8,))
            self.obs_norm_const = np.array([10000, 10000, 10000, 10000, 100, 100, 100, 100], dtype=np.float64)

        elif self.config['mode'] == 'polar':
            self.observation_space = Box(low=-1, high=1, shape=(12,))
            self.obs_norm_const = np.array([10000, 1, 1, 10000, 1, 1, 100, 1, 1, 100, 1, 1], dtype=np.float64)

    def reset(self):
        pass

    def gen_obs(self, env_objs):
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
                raise ValueError('Invalid obs reference {} for obs mode rect'.format(self.config['reference']))

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
                raise ValueError('Invalid obs reference {} for obs mode polar'.format(self.config['reference']))

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

        self.step_reward = 0
        self.total_reward = 0
        self.reward_component_totals = {
            'rejoin': 0,
            'rejoin_first_time': 0,
            'time': 0,
            'distance_change': 0,
            'success': 0,
            'failure': 0,
        }

        self.in_rejoin_prev = False
        self.rejoin_first_time_applied = False

    def _generate_info(self):
        info = {
            'step': self.step_reward,
            'component_totals': self.reward_component_totals,
            'total': self.total_reward
        }

        return info

    def gen_reward(self, env_objs, timestep, status_dict):
        reward = 0

        rejoin_reward = 0
        rejoin_first_time_reward = 0
        time_reward = 0
        distance_change_reward = 0
        failure_reward = 0
        success_reward = 0

        time_reward += self.config['time_decay']

        in_rejoin = status_dict['in_rejoin']

        # compute distance changed between this timestep and previous
        cur_distance = distance2d(env_objs['wingman'], env_objs['rejoin_region'])
        dist_change = cur_distance - self.prev_distance
        self.prev_distance = cur_distance

        if in_rejoin:
            rejoin_reward += self.config['rejoin_timestep'] * timestep

            if not self.rejoin_first_time_applied:
                rejoin_first_time_reward += self.config['rejoin_first_time']
                self.rejoin_first_time_applied = True
        else:
            distance_change_reward += dist_change*self.config['dist_change']

            # if rejoin region is left, refund all accumulated rejoin reward
            #   this is to ensure that the agent doesn't infinitely enter and leave rejoin region
            if self.in_rejoin_prev:
                rejoin_reward += -1*self.reward_component_totals['rejoin']

        self.in_rejoin_prev = in_rejoin

        if status_dict['failure']:
            failure_reward += self.config['failure'][status_dict['failure']]
        elif status_dict['success']:
            success_reward += self.config['success']

        reward += rejoin_reward
        reward += rejoin_first_time_reward
        reward += time_reward
        reward += distance_change_reward
        reward += success_reward
        reward += failure_reward

        self.step_reward = reward
        self.total_reward += reward
        self.reward_component_totals['rejoin'] += rejoin_reward
        self.reward_component_totals['rejoin_first_time'] += rejoin_first_time_reward
        self.reward_component_totals['time'] += time_reward
        self.reward_component_totals['distance_change'] += distance_change_reward
        self.reward_component_totals['success'] += success_reward
        self.reward_component_totals['failure'] += failure_reward

        return reward

class DubinsConstraintIntegration():
    def __init__(self, config):
        self.config = config
        self.reset()
    
    def reset(self):
        self.time_elapsed = 0
        self.rejoin_time = 0
        self.in_rejoin = False

    def step(self, env_objs, timestep):
        # increment rejoin time
        in_rejoin = self.check_rejoin_cond(env_objs)
        if in_rejoin:
            self.rejoin_time += timestep
        else:
            self.rejoin_time = 0

        self.time_elapsed += timestep

        return self.check_constraints(env_objs)
    
    def check_constraints(self, env_objs):
        # get rejoin status
        in_rejoin = self.check_rejoin_cond(env_objs)

        # check success/failure conditions
        lead_distance =  distance2d(env_objs['wingman'], env_objs['lead'])
        
        if lead_distance < self.config['safety_margin']['aircraft']:
            failure = 'crash'
        elif self.time_elapsed > self.config['timeout']:
            failure = 'timeout'
        elif lead_distance >= self.config['max_goal_distance']:
            failure = 'distance'
        else:
            failure = False

        if self.rejoin_time > self.config['success']['rejoin_time']:
            success = True
        else:
            success = False

        status_dict = {
            'success': success,
            'failure': failure,
            'in_rejoin': in_rejoin,
            'time_elapsed': self.time_elapsed
        }

        return status_dict

    def check_rejoin_cond(self, env_objs):
        wingman_coords = (env_objs['wingman'].x, env_objs['wingman'].y)
        return env_objs['rejoin_region'].contains(wingman_coords)
