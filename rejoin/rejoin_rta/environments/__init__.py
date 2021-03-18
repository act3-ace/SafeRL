import math
import numpy as np
import random

import gym

from rejoin_rta.utils.util import draw_from_rand_bounds_dict

class BaseEnv(gym.Env):
    def __init__(self, config):
        # save config
        self.config = config

        if 'verbose' in config:
            self.verbose = config['verbose']
        else:
            self.verbose = False

        self.obs_processor = self.config['obs']['processor'](self.config['obs'])
        self.reward_processor = self.config['reward']['processor'](self.config["reward"])
        self.constraints_processor = self.config['constraints']['processor'](self.config['constraints'])

        self._setup_env_objs()
        self._setup_action_space()
        self._setup_obs_space()

        self.timestep = 1 #TODO

        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        # note that python random should not be used (use numpy random instead)
        # Setting seed just to be safe incase it is accidentally used
        random.seed(seed)

        return [seed]
    
    def step(self, action):
        self._step_sim(action)

        self.status_dict = self._generate_constraint_status()

        reward = self._generate_reward()
        obs = self._generate_obs()
        info = self._generate_info()

        # determine if done
        if self.status_dict['success'] or self.status_dict['failure']:
            done = True
        else:
            done = False

        return  obs, reward, done, info

    def _step_sim(self, action):
        raise NotImplementedError

    def reset(self):
        # apply random initilization to environment objects
        init_dict = self.config['init']

        successful_init = False
        while not successful_init:
            init_dict_draw = draw_from_rand_bounds_dict(init_dict)
            for obj_key, obj_init_dict in init_dict_draw.items():
                self.env_objs[obj_key].reset(**obj_init_dict)

            # TODO check if initialization is safe
            successful_init = True

        # reset processor objects
        self.reward_processor.reset(self.env_objs)
        self.obs_processor.reset()
        self.constraints_processor.reset()

        # reset status dict
        self.status_dict = self.constraints_processor.check_constraints(self.env_objs)

        # generate reset state observations
        obs = self._generate_obs()

        if self.verbose:
            print("env reset with params {}".format(self._generate_info()))

        return obs

    def _setup_env_objs(self):
        self.env_objs = {}
        self.agent = None
        raise NotImplementedError

    def _setup_obs_space(self):
        self.observation_space = self.obs_processor.observation_space

    def _setup_action_space(self):
        self.action_space = self.agent.action_space
        
    def _generate_obs(self):
        obs = self.obs_processor.gen_obs(self.env_objs)
        return obs

    def _generate_reward(self):
        reward = self.reward_processor.gen_reward(self.env_objs, self.timestep, self.status_dict)
        return reward

    def _generate_constraint_status(self):
        return self.constraints_processor.step(self.env_objs, self.timestep)

    def _generate_info(self):
        info = {
            'failure': self.status_dict['failure'],
            'success': self.status_dict['success'],
            'status': self.status_dict,
            'reward': self.reward_processor._generate_info(),
        }

        return info