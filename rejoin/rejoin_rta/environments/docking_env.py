import math
import numpy as np
import random

import gym
from gym.spaces import Discrete, Box

from rejoin.rejoin_rta.environments import BaseEnv, RewardManager
from rejoin.rejoin_rta.aero_models.cwh_spacecraft import CWHSpacecraft
from rejoin.rejoin_rta.utils.geometry import RelativeCircle2d, distance, RelativeCylinder


class DockingEnv(BaseEnv):

    def __init__(self, config):
        super(DockingEnv, self).__init__(config)
        self.timestep = 1

    def _setup_env_objs(self):
        deputy = CWHSpacecraft(config=self.config['agent'])
        chief = CWHSpacecraft()

        if self.config['docking_region']['type'] == 'circle':
            radius = self.config['docking_region']['radius']
            docking_region = RelativeCircle2d(chief, radius=radius, x_offset=0, y_offset=0)
        elif self.config['docking_region']['type'] == 'cylinder':
            docking_region = RelativeCylinder(chief, x_offset=0, y_offset=0, z_offset=0, **self.config['docking_region']['params'])
        else:
            raise ValueError('Invalid docking region type {} not supported'.format(self.config['docking_region']['type']))

        self.env_objs = {
            'deputy': deputy,
            'chief': chief,
            'docking_region': docking_region,
        }

        self.agent = deputy

    def reset(self):
        return super(DockingEnv, self).reset()

    def _step_sim(self, action):
        self.env_objs['chief'].step(self.timestep)
        self.env_objs['deputy'].step(self.timestep, action)

    def _generate_info(self):
        info = {
            'deputy': self.env_objs['deputy']._generate_info(),
            'chief': self.env_objs['chief']._generate_info(),
            'docking_region': self.env_objs['docking_region']._generate_info(),
            'failure': self.status_dict['failure'],
            'success': self.status_dict['success'],
            'status': self.status_dict,
            'reward': self.reward_processor._generate_info(),
        }

        return info


class DockingObservationProcessor:
    def __init__(self, config):
        self.config = config

        low = np.finfo(np.float32).min
        high = np.finfo(np.float32).max

        if self.config['mode'] == '2d':
            self.observation_space = Box(low=low, high=high, shape=(4,))
        elif self.config['mode'] == '3d':
            self.observation_space = Box(low=low, high=high, shape=(6,))
        else:
            raise ValueError("Invalid observation mode {}. Should be one of ".format(self.config['mode']))

    def reset(self):
        pass

    def gen_obs(self, env_objs):
        if self.config['mode'] == '2d':
            obs = env_objs['deputy'].state2d
        elif self.config['mode'] == '3d':
            obs = env_objs['deputy'].state

        return obs


class DockingRewardProcessor:
    def __init__(self, config):
        self.config = config

    def reset(self, env_objs):
        self.prev_distance = distance(env_objs['deputy'], env_objs['docking_region'])

        self.step_reward = 0
        self.total_reward = 0
        self.reward_component_totals = {
            'time': 0,
            'distance_change': 0,
            'success': 0,
            'failure': 0,
        }

    def _generate_info(self):
        info = {
            'step': self.step_reward,
            'component_totals': self.reward_component_totals,
            'total': self.total_reward
        }

        return info

    def gen_reward(self, env_objs, timestep, status_dict):
        reward = 0

        time_reward = 0
        distance_change_reward = 0
        failure_reward = 0
        success_reward = 0

        time_reward += self.config['time_decay']

        # compute distance changed between this timestep and previous
        cur_distance = distance(env_objs['deputy'], env_objs['docking_region'])
        dist_change = cur_distance - self.prev_distance
        self.prev_distance = cur_distance
        distance_change_reward += dist_change*self.config['dist_change']

        if status_dict['failure']:
            failure_reward += self.config['failure'][status_dict['failure']]
        elif status_dict['success']:
            success_reward += self.config['success']

        reward += time_reward
        reward += distance_change_reward
        reward += success_reward
        reward += failure_reward

        self.step_reward = reward
        self.total_reward += reward
        self.reward_component_totals['time'] += time_reward
        self.reward_component_totals['distance_change'] += distance_change_reward
        self.reward_component_totals['success'] += success_reward
        self.reward_component_totals['failure'] += failure_reward

        return reward


class DockingRewardProcessor3D:
    def __init__(self, config):
        self.config = config

    def reset(self, env_objs):
        self.prev_distance = distance(env_objs['deputy'], env_objs['docking_region'])
        self.prev_distance_z = abs(env_objs['deputy'].z - env_objs['docking_region'].z)

        self.step_reward = 0
        self.total_reward = 0
        self.reward_component_totals = {
            'time': 0,
            'distance_change': 0,
            'distance_change_z': 0,
            'success': 0,
            'failure': 0,
        }

    def _generate_info(self):
        info = {
            'step': self.step_reward,
            'component_totals': self.reward_component_totals,
            'total': self.total_reward
        }

        return info

    def gen_reward(self, env_objs, timestep, status_dict):
        reward = 0

        time_reward = 0
        distance_change_reward = 0
        distance_change_z_reward = 0
        failure_reward = 0
        success_reward = 0

        time_reward += self.config['time_decay']

        # compute distance changed between this timestep and previous
        cur_distance = distance(env_objs['deputy'], env_objs['docking_region'])
        dist_change = cur_distance - self.prev_distance
        self.prev_distance = cur_distance
        distance_change_reward += dist_change*self.config['dist_change']

        cur_distance_z = abs(env_objs['deputy'].z - env_objs['docking_region'].z)
        dist_z_change = cur_distance_z - self.prev_distance_z
        self.prev_distance_z = cur_distance_z
        distance_change_z_reward += dist_z_change*self.config['dist_z_change']


        if status_dict['failure']:
            failure_reward += self.config['failure'][status_dict['failure']]
        elif status_dict['success']:
            success_reward += self.config['success']

        reward += time_reward
        reward += distance_change_reward
        reward += distance_change_z_reward
        reward += success_reward
        reward += failure_reward

        self.step_reward = reward
        self.total_reward += reward
        self.reward_component_totals['time'] += time_reward
        self.reward_component_totals['distance_change'] += distance_change_reward
        self.reward_component_totals['distance_change_z'] += distance_change_z_reward
        self.reward_component_totals['success'] += success_reward
        self.reward_component_totals['failure'] += failure_reward

        return reward


class DockingRewardProcessor3Dv2:
    def __init__(self, config):
        self.config = config
        self.prev_distance = 0
        self.prev_distance_z = 0

        self.reward_manager = RewardManager(config=config)

    def reset(self, env_objs):
        self.prev_distance = distance(env_objs['deputy'], env_objs['docking_region'])
        self.prev_distance_z = abs(env_objs['deputy'].z - env_objs['docking_region'].z)

        self.reward_manager.reset(env_objs=env_objs)

    def _generate_info(self):
        info = {
            'step': self.reward_manager.step_value,
            'component_totals': self.reward_manager.components,
            'total': self.reward_manager.total_value,
        }

        return info

    def gen_reward(self, env_objs, timestep, status_dict):
        # --- Preprocess step rewards ---
        step_rewards = self.reward_manager.step(status=status_dict)

        # --- Modify step rewards based on environment ---

        # compute distance changed between this timestep and previous
        cur_distance = distance(env_objs['deputy'], env_objs['docking_region'])
        dist_change = cur_distance - self.prev_distance
        self.prev_distance = cur_distance
        step_rewards['dist_change'] += dist_change*step_rewards['dist_change']

        cur_distance_z = abs(env_objs['deputy'].z - env_objs['docking_region'].z)
        dist_z_change = cur_distance_z - self.prev_distance_z
        self.prev_distance_z = cur_distance_z
        step_rewards['dist_z_change'] += dist_z_change*step_rewards['dist_z_change']

        # Set success/failure reward
        step_rewards['success'] = step_rewards['success'] * status_dict['success']
        step_rewards['failure'] = step_rewards['failure'] * status_dict['failure']

        # --- Update reward manager with step rewards ---
        self.reward_manager.update(step_reward=step_rewards)

        return self.reward_manager.step_reward


class DockingConstraintProcessor:
    def __init__(self, config):
        self.config = config
        self.reset()
    
    def reset(self):
        self.time_elapsed = 0
        self.in_docking = False

    def step(self, env_objs, timestep):
        self.time_elapsed += timestep

        return self.check_constraints(env_objs)
    
    def check_constraints(self, env_objs):
        # get docking status
        in_docking = self.check_docking_cond(env_objs)

        # check success/failure conditions
        dock_distance =  distance(env_objs['deputy'], env_objs['docking_region'])
        
        if self.time_elapsed > self.config['timeout']:
            failure = 'timeout'
        elif dock_distance >= self.config['max_goal_distance']:
            failure = 'distance'
        else:
            failure = False

        if in_docking:
            success = True
        else:
            success = False

        status_dict = {
            'success': success,
            'failure': failure,
            'in_docking': in_docking,
            'time_elapsed': self.time_elapsed
        }

        return status_dict

    def check_docking_cond(self, env_objs):
        return env_objs['docking_region'].contains(env_objs['deputy'])
