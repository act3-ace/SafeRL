import gym.spaces
import numpy as np

from saferl.environment.tasks import ObservationProcessor, RewardProcessor, StatusProcessor
from saferl.environment.models import distance


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

    def generate_observation(self, env_objs):
        obs = env_objs['deputy'].state.vector
        return obs

    def _increment(self, env_objs, timestep, status):
        # update status and return
        return self.generate_observation(env_objs)

    def _process(self, env_objs, status):
        # return the current status
        return self.generate_observation(env_objs)


class TimeRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)

    # def generate_reward(self, env_objs, timestep, status):
    #     step_reward = self.reward
    #     return step_reward

    def _increment(self, env_objs, timestep, status):
        # return step_value of reward accumulated during interval of size timestep
        step_reward = self.reward
        return step_reward

    def _process(self, env_objs, status):
        # return the current state of reward component
        return self.total_value


class DistanceChangeRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.prev_distance = 0
        self.deputy = self.config["deputy"]
        self.docking_region = self.config["docking_region"]

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.prev_distance = distance(env_objs[self.deputy], env_objs[self.docking_region])

    # def generate_reward(self, env_objs, timestep, status):
    #     cur_distance = distance(env_objs[self.docking_region], env_objs[self.docking_region])
    #     dist_change = cur_distance - self.prev_distance
    #     self.prev_distance = cur_distance
    #     step_reward = dist_change * self.reward
    #     return step_reward

    def _increment(self, env_objs, timestep, status):
        # return step_value of reward accumulated during interval of size timestep
        cur_distance = distance(env_objs[self.docking_region], env_objs[self.docking_region])
        dist_change = cur_distance - self.prev_distance
        self.prev_distance = cur_distance
        step_reward = dist_change * self.reward
        return step_reward

    def _process(self, env_objs, status):
        # return the current state of reward component
        return self.total_value


class DistanceChangeZRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.prev_distance = 0
        self.deputy = self.config["deputy"]
        self.docking_region = self.config["docking_region"]

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.prev_distance = abs(env_objs[self.deputy].z - env_objs[self.docking_region].z)

    # def generate_reward(self, env_objs, timestep, status):
    #     cur_distance_z = abs(env_objs[self.deputy].z - env_objs[self.docking_region].z)
    #     dist_z_change = cur_distance_z - self.prev_distance
    #     self.prev_distance = cur_distance_z
    #     step_reward = dist_z_change * self.reward
    #     return step_reward

    def _increment(self, env_objs, timestep, status):
        # return step_value of reward accumulated during interval of size timestep
        cur_distance_z = abs(env_objs[self.deputy].z - env_objs[self.docking_region].z)
        dist_z_change = cur_distance_z - self.prev_distance
        self.prev_distance = cur_distance_z
        step_reward = dist_z_change * self.reward
        return step_reward

    def _process(self, env_objs, status):
        # return the current state of reward component
        return self.total_value


class SuccessRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.success_status = self.config["success_status"]

    # def generate_reward(self, status):
    #     step_reward = 0
    #     if status[self.success_status]:
    #         step_reward = self.reward
    #     return step_reward

    def _increment(self, env_objs, timestep, status):
        # return step_value of reward accumulated during interval of size timestep
        cur_distance_z = abs(env_objs[self.deputy].z - env_objs[self.docking_region].z)
        dist_z_change = cur_distance_z - self.prev_distance
        self.prev_distance = cur_distance_z
        step_reward = dist_z_change * self.reward
        return step_reward

    def _process(self, env_objs, status):
        # return the current state of reward component
        return self.total_value


class FailureRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.failure_status = self.config["failure_status"]

    # def generate_reward(self, env_objs, timestep, status):
    #     step_reward = 0
    #     if status[self.failure_status]:
    #         step_reward = self.reward[status[self.failure_status]]
    #     return step_reward

    def _increment(self, env_objs, timestep, status):
        # return step_value of reward accumulated during interval of size timestep
        step_reward = 0
        if status[self.failure_status]:
            step_reward = self.reward[status[self.failure_status]]
        return step_reward

    def _process(self, env_objs, status):
        # return the current state of reward component
        return self.total_value


class DockingStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.docking_region = self.config["docking_region"]
        self.deputy = self.config["deputy"]

    def generate_status(self, env_objs):
        in_docking = env_objs[self.docking_region].contains(env_objs[self.deputy])
        return in_docking

    def _increment(self, env_objs, timestep, status):
        # update status and return
        return self.generate_status(env_objs)

    def _process(self, env_objs, status):
        # return the current status
        return self.generate_status(env_objs)


class DockingDistanceStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.docking_region = self.config["docking_region"]
        self.deputy = self.config["deputy"]

    def generate_status(self, env_objs):
        docking_distance = distance(env_objs[self.deputy], env_objs[self.docking_region])
        return docking_distance

    def _increment(self, env_objs, timestep, status):
        # update status and return
        return self.generate_status(env_objs)

    def _process(self, env_objs, status):
        # return the current status
        return self.generate_status(env_objs)


class FailureStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.time_elapsed = 0
        self.timeout = self.config["timeout"]
        self.docking_distance = self.config["docking_distance"]
        self.max_goal_distance = self.config["max_goal_distance"]

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.time_elapsed = 0

    def generate_status(self, status):
        if self.time_elapsed > self.timeout:
            failure = 'timeout'
        elif status[self.docking_distance] >= self.max_goal_distance:
            failure = 'distance'
        else:
            failure = False

        return failure

    def _increment(self, env_objs, timestep, status):
        # update status and return
        self.time_elapsed += timestep
        return self.generate_status(status)

    def _process(self, env_objs, status):
        # return the current status
        return self.generate_status(status)


class SuccessStatusProcessor(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.docking_status = self.config["docking_status"]

    def generate_status(self, status):
        success = status[self.docking_status]
        return success

    def _increment(self, env_objs, timestep, status):
        # update status and return
        return self.generate_status(status)

    def _process(self, env_objs, status):
        # return the current status
        return self.generate_status(status)
