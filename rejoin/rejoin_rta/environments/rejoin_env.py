import math
import numpy as np
import random
from scipy.spatial.transform import Rotation


import gym
from gym.spaces import Discrete, Box

from rejoin_rta.environments import BaseEnv
from rejoin_rta.aero_models.dubins import Dubins2dPlatform
from rejoin_rta.utils.geometry import RelativeCircle, distance

class DubinsRejoin(BaseEnv):

    def __init__(self, config):
        super(DubinsRejoin, self).__init__(config)
        self.timestep = 1

    def _setup_env_objs(self):
        wingman = Dubins2dPlatform(controller='agent', config=self.config['agent'])
        lead = Dubins2dPlatform()

        if self.config['rejoin_region']['type'] == 'circle':
            r_offset = self.config['rejoin_region']['range']
            radius = self.config['rejoin_region']['radius']
            aspect_angle = np.radians(self.config['rejoin_region']['aspect_angle'])
            rejoin_region = RelativeCircle(lead, radius=radius, track_orientation=True, r_offset=r_offset, aspect_angle=aspect_angle)
        else:
            raise ValueError('Invalid rejoin region type {} not supported'.format(self.config['rejoin_region']['type']))

        self.env_objs = {
            'wingman': wingman,
            'lead': lead,
            'rejoin_region': rejoin_region,
        }

        self.agent = wingman

    def reset(self):
        return super(DubinsRejoin, self).reset()

    def _step_sim(self, action):
        self.env_objs['lead'].step(self.timestep)
        self.env_objs['wingman'].step(self.timestep, action)

    def _generate_info(self):
        info = {
            'wingman': self.env_objs['wingman']._generate_info(),
            'lead': self.env_objs['lead']._generate_info(),
            'rejoin_region': self.env_objs['rejoin_region']._generate_info(),
            'failure': self.status_dict['failure'],
            'success': self.status_dict['success'],
            'status': self.status_dict,
            'reward': self.reward_processor._generate_info(),
        }

        return info

class DubinsObservationProcessor():
    def __init__(self, config):
        self.config = config

        if self.config['mode'] == 'rect':
            self.observation_space = Box(low=-1, high=1, shape=(8,))
            self.obs_norm_const = np.array([10000, 10000, 10000, 10000, 100, 100, 100, 100], dtype=np.float64)

        elif self.config['mode'] == 'magnorm':
            self.observation_space = Box(low=-1, high=1, shape=(12,))
            self.obs_norm_const = np.array([10000, 1, 1, 10000, 1, 1, 100, 1, 1, 100, 1, 1], dtype=np.float64)

    def reset(self):
        pass

    def gen_obs(self, env_objs):
        def vec2magnorm(vec):
                norm = np.linalg.norm(vec)
                mag_norm_vec = np.concatenate( ( [norm], vec / norm ) )
                return mag_norm_vec

        wingman_lead_r = env_objs['lead'].position - env_objs['wingman'].position
        wingman_rejoin_r = env_objs['rejoin_region'].position - env_objs['wingman'].position

        wingman_vel = env_objs['wingman'].velocity
        lead_vel = env_objs['lead'].velocity

        reference_rotation = Rotation.from_quat([0, 0, 0, 1])
        if self.config['reference'] == 'wingman':
            reference_rotation = env_objs['wingman'].orientation.inv()

        wingman_lead_r = reference_rotation.apply(wingman_lead_r)
        wingman_rejoin_r = reference_rotation.apply(wingman_rejoin_r)

        wingman_vel = reference_rotation.apply(wingman_vel)
        lead_vel = reference_rotation.apply(lead_vel)

        if self.config['mode'] == 'magnorm':
            wingman_lead_r = vec2magnorm(wingman_lead_r)
            wingman_rejoin_r = vec2magnorm(wingman_rejoin_r)

            wingman_vel = vec2magnorm(wingman_vel)
            lead_vel = vec2magnorm(lead_vel)

        obs = np.concatenate( [
            wingman_lead_r[0:3],
            wingman_rejoin_r[0:3],
            wingman_vel[0:3],
            lead_vel[0:3],
        ] )

        # normalize observation
        obs = np.divide(obs, self.obs_norm_const)

        obs = np.clip(obs, -1, 1)

        return obs

class DubinsRewardProcessor():
    def __init__(self, config):
        self.config = config

    def reset(self, env_objs):
        self.prev_distance = distance(env_objs['wingman'], env_objs['rejoin_region'])

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
        cur_distance = distance(env_objs['wingman'], env_objs['rejoin_region'])
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

class DubinsConstraintProcessor():
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
        lead_distance =  distance(env_objs['wingman'], env_objs['lead'])
        
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
        return env_objs['rejoin_region'].contains(env_objs['wingman'])
