import random

import numpy as np

import gym

from saferl.environment.tasks.manager import RewardManager, ObservationManager, StatusManager
from saferl.environment.tasks.initializers import RandBoundsInitializer
from saferl.environment.utils import setup_env_objs_from_config, setup_initializers_from_config


class BaseEnv(gym.Env):

    def __init__(self, config):
        # save config
        self.config = config

        # Initialize sim_state
        agent, env_objs = self._setup_env_objs()
        self.sim_state = SimulationState(agent=agent, env_objs=env_objs)

        # Get initializers
        self.initializers = setup_initializers_from_config(config, env_objs, default_init=RandBoundsInitializer)

        if 'verbose' in config:
            self.verbose = config['verbose']
        else:
            self.verbose = False

        self.observation_manager = ObservationManager(self.config["observation"])
        self.reward_manager = RewardManager(config=self.config["reward"])
        self.status_manager = StatusManager(config=self.config["status"])

        self._setup_action_space()
        self._setup_obs_space()

        self.step_size = 1  # TODO

        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        # note that python random should not be used (use numpy random instead)
        # Setting seed just to be safe in case it is accidentally used
        random.seed(seed)

        return [seed]

    def step(self, action):
        self._step_sim(action)

        self.sim_state.status = self._generate_status()

        reward = self._generate_reward()
        obs = self._generate_obs()
        info = self.generate_info()

        # determine if done
        if self.status['success'] or self.status['failure']:
            done = True
        else:
            done = False

        return obs, reward, done, info

    def _step_sim(self, action):
        raise NotImplementedError

    def reset(self):
        # Reinitialize env_objs
        self._initialize()

        # reset processor objects and status
        self.sim_state.status = self.status_manager.reset(self.sim_state)
        self.reward_manager.reset(self.sim_state)
        self.observation_manager.reset(self.sim_state)

        # generate reset state observations
        obs = self._generate_obs()

        if self.verbose:
            print("env reset with params {}".format(self.generate_info()))

        return obs

    def _initialize(self):
        # Reinitialize env_objs
        for initializer in self.initializers:
            initializer.initialize()

    def _setup_env_objs(self):
        agent, env_objs = setup_env_objs_from_config(self.config)
        return agent, env_objs

    def _setup_obs_space(self):
        self.observation_space = self.observation_manager.observation_space

    def _setup_action_space(self):
        self.action_space = self.agent.action_space

    def _generate_obs(self):
        # TODO: Handle multiple observations
        self.observation_manager.step(
            self.sim_state,
            self.step_size,
        )
        return self.observation_manager.obs

    def _generate_reward(self):
        self.reward_manager.step(
            self.sim_state,
            self.step_size,
        )
        return self.reward_manager.step_value

    def _generate_status(self):

        status = self.status_manager.step(
            self.sim_state,
            self.step_size,
        )

        return status

    def generate_info(self):
        info = {
            'failure': self.status['failure'],
            'success': self.status['success'],
            'status': self.status,
            'reward': self.reward_manager.generate_info(),
        }

        return info

    @property
    def env_objs(self):
        return self.sim_state.env_objs

    @env_objs.setter
    def env_objs(self, val):
        self.sim_state.env_objs = val

    @property
    def status(self):
        return self.sim_state.status

    @status.setter
    def status(self, val):
        self.sim_state.status = val

    @property
    def agent(self):
        return self.sim_state.agent

    @agent.setter
    def agent(self, val):
        self.sim_state.agent = val


class SimulationState:

    def __init__(self, env_objs=None, agent=None, status=None):
        self.env_objs = env_objs
        self.agent = agent
        self.status = status
