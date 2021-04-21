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

    def _process(self, env_objs, status):
        obs = env_objs['deputy'].state.vector
        return obs


class TimeRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)

    def reset(self, env_objs, status):
        super().reset(env_objs=env_objs, status=status)
        self.previous_step_size = 0


    def _increment(self, env_objs, step_size, status):
        # update state
        self.previous_step_size = step_size

    def _process(self, env_objs, status):
        step_reward = self.previous_step_size * self.reward
        return step_reward


class DistanceChangeRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.deputy = self.config["deputy"]
        self.docking_region = self.config["docking_region"]

    def reset(self, env_objs, status):
        super().reset(env_objs=env_objs, status=status)
        self.cur_distance = distance(env_objs[self.deputy], env_objs[self.docking_region])
        self.prev_distance = distance(env_objs[self.deputy], env_objs[self.docking_region])

    def _increment(self, env_objs, step_size, status):
        self.prev_distance = self.cur_distance
        self.cur_distance = distance(env_objs[self.deputy], env_objs[self.docking_region])

    def _process(self, env_objs, status):
        dist_change = self.cur_distance - self.prev_distance
        step_reward = dist_change * self.reward
        return step_reward


class DistanceChangeZRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.deputy = self.config["deputy"]
        self.docking_region = self.config["docking_region"]

    def reset(self, env_objs, status):
        super().reset(env_objs=env_objs, status=status)
        self.prev_z_distance = 0
        self.cur_z_distance = abs(env_objs[self.deputy].z - env_objs[self.docking_region].z)

    def _increment(self, env_objs, step_size, status):
        self.prev_z_distance = self.cur_z_distance
        self.cur_z_distance = abs(env_objs[self.deputy].z - env_objs[self.docking_region].z)

    def _process(self, env_objs, status):
        dist_z_change = self.cur_z_distance - self.prev_z_distance
        step_reward = dist_z_change * self.reward
        return step_reward


class SuccessRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.success_status = self.config["success_status"]

    def _increment(self, env_objs, step_size, status):
        # reward derived straight from status dict, therefore no state machine necessary
        ...

    def _process(self, env_objs, status):
        step_reward = 0
        if status[self.success_status]:
            step_reward = self.reward
        return step_reward


class FailureRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.failure_status = self.config["failure_status"]

    def _increment(self, env_objs, step_size, status):
        # reward derived straight from status dict, therefore no state machine necessary
        ...

    def _process(self, env_objs, status):
        step_reward = 0
        if status[self.failure_status]:
            step_reward = self.reward[status[self.failure_status]]
        return step_reward


class DockingStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.docking_region = self.config["docking_region"]
        self.deputy = self.config["deputy"]

    def reset(self, env_objs, status):
        ...

    def _increment(self, env_objs, step_size, status):
        # status derived directly from simulation state. No state machine necessary
        ...

    def _process(self, env_objs, status):
        in_docking = env_objs[self.docking_region].contains(env_objs[self.deputy])
        return in_docking


class DockingDistanceStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.docking_region = self.config["docking_region"]
        self.deputy = self.config["deputy"]

    def reset(self, env_objs, status):
        ...

    def _increment(self, env_objs, step_size, status):
        # status derived directly from simulation state. No state machine necessary
        ...

    def _process(self, env_objs, status):
        docking_distance = distance(env_objs[self.deputy], env_objs[self.docking_region])
        return docking_distance


class FailureStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.timeout = self.config["timeout"]
        self.docking_distance = self.config["docking_distance"]
        self.max_goal_distance = self.config["max_goal_distance"]

    def reset(self, env_objs, status):
        self.time_elapsed = 0

    def _increment(self, env_objs, step_size, status):
        # increment internal state
        self.time_elapsed += step_size

    def _process(self, env_objs, status):
        # process state and return status
        if self.time_elapsed > self.timeout:
            failure = 'timeout'
        elif status[self.docking_distance] >= self.max_goal_distance:
            failure = 'distance'
        else:
            failure = False

        return failure


class SuccessStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.docking_status = self.config["docking_status"]

    def reset(self, env_objs, status):
        ...

    def _increment(self, env_objs, step_size, status):
        # status derived directly from simulation state, therefore no state machine needed
        ...

    def _process(self, env_objs, status):
        # process stare and return status
        success = status[self.docking_status]
        return success