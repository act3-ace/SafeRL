import random
import numpy as np
import gymnasium as gym
from typing import Any
import copy

from saferl.environment.tasks.manager import RewardManager, ObservationManager, StatusManager
from saferl.environment.tasks.processor.status import TimeoutStatusProcessor, NeverSuccessStatusProcessor
from saferl.environment.utils import setup_env_objs_from_config, recursive_np_copy
from saferl.environment.constants import STATUS, REWARD, OBSERVATION, VERBOSE, RENDER
from saferl.environment.tasks.initializers import RandBoundsInitializer
from saferl.environment.models.platforms import BasePlatform


class BaseEnv(gym.Env):

    def __init__(self, env_config):

        # Set time step size
        if 'step_size' in env_config:
            self.step_size = env_config['step_size']
        else:
            self.step_size = 1

        # Initialize simulation state
        self.sim_state = SimulationState()

        # Set verbosity level
        if VERBOSE in env_config.keys():
            self.verbose = env_config[VERBOSE]
        else:
            self.verbose = False

        # Create managers
        self.observation_manager = ObservationManager(env_config[OBSERVATION])
        self.reward_manager = RewardManager(env_config[REWARD])
        self.status_manager = StatusManager(env_config[STATUS])

        # Create renderer
        if RENDER in env_config.keys():
            render_class = env_config[RENDER]["class"] if "class" in env_config[RENDER].keys() else None
            self.render_config = env_config[RENDER]["config"] if "config" in env_config[RENDER].keys() else {}
            if render_class is not None:
                self.renderer = render_class(**self.render_config)
        else:
            self.render_config = {}
            self.renderer = None

        # Create default success/failure processors
        has_failure_processor = False
        has_success_processor = False

        for processor in self.status_manager.processors:
            if processor.name == 'failure':
                has_failure_processor = True
            elif processor.name == 'success':
                has_success_processor = True

        if not has_failure_processor:
            self.status_manager.processors.append(TimeoutStatusProcessor())
        if not has_success_processor:
            self.status_manager.processors.append(NeverSuccessStatusProcessor())

        # Get environment objects and initializers
        self.sim_state.agent, self.sim_state.env_objs, self.initializers = setup_env_objs_from_config(
            config=env_config,
            default_initializer=RandBoundsInitializer)

        self.last_info: dict
        self.last_obs: np.ndarray
        self.last_action: Any

        # Setup action and observation space
        self._setup_action_space()
        self._setup_obs_space()

        # Reset environment
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        # note that python random should not be used (use numpy random instead)
        # Setting seed just to be safe in case it is accidentally used
        random.seed(seed)

        return [seed]

    def _step_sim(self, action):
        agent_name = self.sim_state.agent.name
        platforms = [obj_item for obj_item in self.sim_state.env_objs.items() if isinstance(obj_item[1], BasePlatform)]
        for obj_name, obj in platforms:
            if obj_name == agent_name:
                self.sim_state.env_objs[obj_name].step_compute(self.sim_state, self.step_size, action)
            else:
                self.sim_state.env_objs[obj_name].step_compute(self.sim_state, self.step_size)

        for obj_name, obj in platforms:
            self.sim_state.env_objs[obj_name].step_apply()

    def step(self, action):

        self._step_sim(action)

        # update time metrics - timesteps and time_elapsed
        self.time_elapsed += self.step_size
        self.timesteps_elapsed += 1

        # update status and generate logs
        self.sim_state.status = self._generate_status()
        reward = self._generate_reward()
        obs = self._generate_obs()
        info = self.generate_info()

        terminated = False
        truncated = False  # todo set truncated if timeout
        # determine if done
        if self.status['success'] or self.status['failure']:
            terminated = True

        self.last_info = recursive_np_copy(info)
        self.last_obs = recursive_np_copy(obs)
        self.last_action = recursive_np_copy(action)

        return obs, reward, terminated, truncated, {}

    def reset(self, *, seed=None, options=None):
        # Reinitialize env_objs
        self._initialize()

        # reset processor objects and status
        self.sim_state.reset()
        self.sim_state.status = self.status_manager.reset(self.sim_state)
        self.reward_manager.reset(self.sim_state)
        self.observation_manager.reset(self.sim_state)

        info = self.generate_info()

        # generate reset state observations
        obs = self._generate_obs()
        self.last_info = recursive_np_copy(info)
        self.last_obs = recursive_np_copy(obs)
        self.last_action = None

        # Reset render viewer
        if self.renderer is not None:
            self.renderer.reset()

        if self.verbose:
            print("env reset with params {}".format(self.generate_info()))

        return obs, {}

    def render(self, mode='human'):
        if self.renderer is not None:
            self.renderer.render(state=self.sim_state)

    def _initialize(self):
        # Reinitialize env_objs
        for initializer in self.initializers:
            initializer.initialize()

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
            'timestep_size': self.step_size,
            'timesteps_elapsed': self.timesteps_elapsed,
            'time_elapsed': self.time_elapsed
        }

        for obj_name in self.env_objs:
            info[obj_name] = self.env_objs[obj_name].generate_info()

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

    @property
    def time_elapsed(self):
        return self.sim_state.time_elapsed

    @time_elapsed.setter
    def time_elapsed(self, val):
        self.sim_state.time_elapsed = val

    @property
    def timesteps_elapsed(self):
        return self.sim_state.timesteps_elapsed

    @timesteps_elapsed.setter
    def timesteps_elapsed(self, val):
        self.sim_state.timesteps_elapsed = val


class SimulationState:
    def __init__(self, env_objs=None, agent=None, status=None):
        self.env_objs = env_objs
        self.agent = agent
        self.status = status
        self.time_elapsed = 0
        self.timesteps_elapsed = 0

    def reset(self):
        self.status = None
        self.time_elapsed = 0
        self.timesteps_elapsed = 0
