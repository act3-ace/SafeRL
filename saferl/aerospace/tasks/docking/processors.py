import gym.spaces
import numpy as np

from saferl.environment.tasks import ObservationProcessor, RewardProcessor, StatusProcessor
from saferl.environment.models import distance


class DockingObservationProcessor(ObservationProcessor):
    def __init__(self, config):
        super().__init__(config=config)

        low = np.finfo(np.float32).min
        high = np.finfo(np.float32).max

        if self.config['mode'] == '2d':
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=(4,))
        elif self.config['mode'] == '3d':
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=(6,))
        else:
            raise ValueError("Invalid observation mode {}. Should be one of ".format(self.config['mode']))

    def generate_observation(self, env_objs):
        obs = env_objs['deputy'].state.vector
        return obs


class TimeRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)

    def generate_reward(self, env_objs, timestep, status):
        step_reward = self.reward
        return step_reward


class DistanceChangeRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.prev_distance = 0

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.prev_distance = distance(env_objs['deputy'], env_objs['docking_region'])

    def generate_reward(self, env_objs, timestep, status):
        cur_distance = distance(env_objs['deputy'], env_objs['docking_region'])
        dist_change = cur_distance - self.prev_distance
        self.prev_distance = cur_distance
        step_reward = dist_change * self.reward
        return step_reward


class DistanceChangeZRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.prev_distance = 0

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.prev_distance = abs(env_objs['deputy'].z - env_objs['docking_region'].z)

    def generate_reward(self, env_objs, timestep, status):
        cur_distance_z = abs(env_objs['deputy'].z - env_objs['docking_region'].z)
        dist_z_change = cur_distance_z - self.prev_distance
        self.prev_distance = cur_distance_z
        step_reward = dist_z_change * self.reward
        return step_reward


class SuccessRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)

    def generate_reward(self, env_objs, timestep, status):
        step_reward = 0
        if status["success"]:
            step_reward = self.reward
        return step_reward


class FailureRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)

    def generate_reward(self, env_objs, timestep, status):
        step_reward = 0
        if status["failure"]:
            step_reward = self.reward[status['failure']]
        return step_reward


class DockingStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)

    def generate_status(self, env_objs, timestep, status, old_status):
        in_docking = env_objs['docking_region'].contains(env_objs['deputy'])
        return in_docking


class DockingDistanceStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)

    def generate_status(self, env_objs, timestep, status, old_status):
        docking_distance = distance(env_objs['deputy'], env_objs['docking_region'])
        return docking_distance


class FailureStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.time_elapsed = 0

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.time_elapsed = 0

    def generate_status(self, env_objs, timestep, status, old_status):
        self.time_elapsed += timestep

        if self.time_elapsed > self.config['timeout']:
            failure = 'timeout'
        elif status['docking_distance'] >= self.config['max_goal_distance']:
            failure = 'distance'
        else:
            failure = False

        return failure


class SuccessStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)

    def generate_status(self, env_objs, timestep, status, old_status):
        success = status["docking_status"]
        return success
