import gym.spaces
import numpy as np

from saferl.environment.tasks.processor import ObservationProcessor, RewardProcessor, StatusProcessor
from saferl.environment.models.geometry import distance


class DockingObservationProcessor(ObservationProcessor):
    def __init__(self, config):
        super().__init__(config=config)

        # Initialize member variables from config
        self.mode = self.config["mode"]
        self.deputy = self.config["deputy"]

        low = np.finfo(np.float32).min
        high = np.finfo(np.float32).max

        if self.config['mode'] == '2d':
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=(4,))
        elif self.config['mode'] == '3d':
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=(6,))
        else:
            raise ValueError("Invalid observation mode {}. Should be one of ".format(self.config['mode']))

    def _process(self, sim_state):
        obs = sim_state.env_objs['deputy'].state.vector
        return obs

class DockingObservationProcessorOriented(ObservationProcessor):
    def __init__(self, config):
        super().__init__(config=config)

        # Initialize member variables from config
        self.mode = self.config["mode"]
        self.deputy = self.config["deputy"]

        low = np.finfo(np.float32).min
        high = np.finfo(np.float32).max

        if self.config['mode'] == '2d':
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=(7,))
            self.norm_const = np.array([1000, 1000, np.pi, 100, 100, 0.4, 500])
        elif self.config['mode'] == '3d':
            raise NotImplementedError
        else:
            raise ValueError("Invalid observation mode {}. Should be one of ".format(self.config['mode']))

    def _process(self, sim_state):
        obs = sim_state.env_objs['deputy'].state.vector

        # if self.config['mode'] == '2d':
        #     obs[2]

        obs = obs / self.norm_const
        return obs


class TimeRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)

    def reset(self, sim_state):
        super().reset(sim_state)
        self.previous_step_size = 0


    def _increment(self, sim_state, step_size):
        # update state
        self.previous_step_size = step_size

    def _process(self, sim_state):
        step_reward = self.previous_step_size * self.reward
        return step_reward


class DistanceChangeRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.deputy = self.config["deputy"]
        self.docking_region = self.config["docking_region"]

    def reset(self, sim_state):
        super().reset(sim_state)
        self.cur_distance = distance(sim_state.env_objs[self.deputy], sim_state.env_objs[self.docking_region])
        self.prev_distance = distance(sim_state.env_objs[self.deputy], sim_state.env_objs[self.docking_region])

    def _increment(self, sim_state, step_size):
        self.prev_distance = self.cur_distance
        self.cur_distance = distance(sim_state.env_objs[self.deputy], sim_state.env_objs[self.docking_region])

    def _process(self, sim_state):
        dist_change = self.cur_distance - self.prev_distance
        step_reward = dist_change * self.reward
        return step_reward


class DistanceChangeZRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.deputy = self.config["deputy"]
        self.docking_region = self.config["docking_region"]

    def reset(self, sim_state):
        super().reset(sim_state)
        self.prev_z_distance = 0
        self.cur_z_distance = abs(sim_state.env_objs[self.deputy].z - sim_state.env_objs[self.docking_region].z)

    def _increment(self, sim_state, step_size):
        self.prev_z_distance = self.cur_z_distance
        self.cur_z_distance = abs(sim_state.env_objs[self.deputy].z - sim_state.env_objs[self.docking_region].z)

    def _process(self, sim_state):
        dist_z_change = self.cur_z_distance - self.prev_z_distance
        step_reward = dist_z_change * self.reward
        return step_reward


class SuccessRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.success_status = self.config["success_status"]

    def _increment(self, sim_state, step_size):
        # reward derived straight from status dict, therefore no state machine necessary
        ...

    def _process(self, sim_state):
        step_reward = 0
        if sim_state.status[self.success_status]:
            step_reward = self.reward
        return step_reward


class FailureRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.failure_status = self.config["failure_status"]

    def _increment(self, sim_state, step_size):
        # reward derived straight from status dict, therefore no state machine necessary
        ...

    def _process(self, sim_state):
        step_reward = 0
        if sim_state.status[self.failure_status]:
            step_reward = self.reward[sim_state.status[self.failure_status]]
        return step_reward


class DockingStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.docking_region = self.config["docking_region"]
        self.deputy = self.config["deputy"]

    def reset(self, sim_state):
        ...

    def _increment(self, sim_state, step_size):
        # status derived directly from simulation state. No state machine necessary
        ...

    def _process(self, sim_state):
        in_docking = sim_state.env_objs[self.docking_region].contains(sim_state.env_objs[self.deputy])
        return in_docking


class DockingDistanceStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.docking_region = self.config["docking_region"]
        self.deputy = self.config["deputy"]

    def reset(self, sim_state):
        ...

    def _increment(self, sim_state, step_size):
        # status derived directly from simulation state. No state machine necessary
        ...

    def _process(self, sim_state):
        docking_distance = distance(sim_state.env_objs[self.deputy], sim_state.env_objs[self.docking_region])
        return docking_distance


class FailureStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.timeout = self.config["timeout"]
        self.docking_distance = self.config["docking_distance"]
        self.max_goal_distance = self.config["max_goal_distance"]

    def reset(self, sim_state):
        self.time_elapsed = 0

    def _increment(self, sim_state, step_size):
        # increment internal state
        self.time_elapsed += step_size

    def _process(self, sim_state):
        # process state and return status
        if self.time_elapsed > self.timeout:
            failure = 'timeout'
        elif sim_state.status[self.docking_distance] >= self.max_goal_distance:
            failure = 'distance'
        else:
            failure = False

        return failure


class SuccessStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.docking_status = self.config["docking_status"]

    def reset(self, sim_state):
        ...

    def _increment(self, sim_state, step_size):
        # status derived directly from simulation state, therefore no state machine needed
        ...

    def _process(self, sim_state):
        # process stare and return status
        success = sim_state.status[self.docking_status]
        return success