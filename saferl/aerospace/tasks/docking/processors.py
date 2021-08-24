import gym.spaces
import numpy as np
from saferl.environment.tasks.processor import ObservationProcessor, RewardProcessor, StatusProcessor
from saferl.environment.models.geometry import distance


# --------------------- Observation Processors ------------------------

class DockingObservationProcessor(ObservationProcessor):
    def __init__(self, name=None, deputy=None, mode='2d', normalization=None, clip=None, post_processors=None):
        super().__init__(name=name, normalization=normalization, clip=clip, post_processors=post_processors)
        # Initialize member variables from config

        # 2d or 3d
        self.mode = mode
        # not platform ref, string for name of deputy
        self.deputy = deputy


        low = np.finfo(np.float32).min
        high = np.finfo(np.float32).max

        if self.mode == '2d':
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=(4,))
        elif self.mode == '3d':
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=(6,))
        else:
            raise ValueError("Invalid observation mode {}. Should be one of ".format(self.mode))

    def _process(self, sim_state):
        obs = sim_state.env_objs[self.deputy].state.vector
        return obs


class DockingObservationProcessorOriented(ObservationProcessor):
    def __init__(self, name=None, deputy=None, mode='2d', normalization=None, clip=None, post_processors=None):
        # Invoke parent's constructor
        super().__init__(name=name, normalization=normalization, clip=clip, post_processors=post_processors)

        # Initialize member variables from config
        self.mode = mode
        self.deputy = deputy

        low = np.finfo(np.float32).min
        high = np.finfo(np.float32).max

        if self.mode == '2d':
            self.observation_space = gym.spaces.Box(low=low, high=high, shape=(7,))
            if not self.has_normalization:
                # if no custom normalization defined
                self._add_normalization([1000, 1000, np.pi, 100, 100, 0.4, 500])
        elif self.mode == '3d':
            raise NotImplementedError
        else:
            raise ValueError("Invalid observation mode {}. Should be one of ".format(self.mode))

    def _process(self, sim_state):
        obs = sim_state.env_objs[self.deputy].state.vector

        # if self.config['mode'] == '2d':
        #     obs[2]

        return obs


# --------------------- Reward Processors ------------------------


class TimeRewardProcessor(RewardProcessor):
    def __init__(self, name=None, reward=None):
        super().__init__(name=name, reward=reward)

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
    def __init__(self, name=None, deputy=None, docking_region=None, reward=None):
        super().__init__(name=name, reward=reward)
        self.deputy = deputy
        self.docking_region = docking_region

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
    def __init__(self, name=None, deputy=None, docking_region=None, reward=None):
        super().__init__(name=name, reward=reward)
        self.deputy = deputy
        self.docking_region = docking_region

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
    def __init__(self, name=None, success_status=None, reward=None):
        super().__init__(name=name, reward=reward)
        self.success_status = success_status

    def _increment(self, sim_state, step_size):
        # reward derived straight from status dict, therefore no state machine necessary
        pass

    def _process(self, sim_state):
        step_reward = 0
        if sim_state.status[self.success_status]:
            step_reward = self.reward
        return step_reward


class FailureRewardProcessor(RewardProcessor):
    def __init__(self, name=None, failure_status=None, reward=None):
        super().__init__(name=name, reward=reward)
        self.failure_status = failure_status

    def _increment(self, sim_state, step_size):
        # reward derived straight from status dict, therefore no state machine necessary
        pass

    def _process(self, sim_state):
        step_reward = 0
        if sim_state.status[self.failure_status]:
            step_reward = self.reward[sim_state.status[self.failure_status]]
        return step_reward


# --------------------- Status Processors ------------------------


class DockingStatusProcessor(StatusProcessor):
    def __init__(self, name=None, deputy=None, docking_region=None):
        super().__init__(name=name)
        self.docking_region = docking_region
        self.deputy = deputy

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        # status derived directly from simulation state. No state machine necessary
        pass

    def _process(self, sim_state):
        in_docking = sim_state.env_objs[self.docking_region].contains(sim_state.env_objs[self.deputy])
        return in_docking


class DockingDistanceStatusProcessor(StatusProcessor):
    def __init__(self, name=None, deputy=None, docking_region=None):
        super().__init__(name=name)
        self.docking_region = docking_region
        self.deputy = deputy

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        # status derived directly from simulation state. No state machine necessary
        pass

    def _process(self, sim_state):
        docking_distance = distance(sim_state.env_objs[self.deputy], sim_state.env_objs[self.docking_region])
        return docking_distance


class FailureStatusProcessor(StatusProcessor):
    def __init__(self, name=None, docking_distance=None, max_goal_distance=None, timeout=None):
        super().__init__(name=name)
        self.timeout = timeout
        self.docking_distance = docking_distance
        self.max_goal_distance = max_goal_distance

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
    def __init__(self, name=None, docking_status=None):
        super().__init__(name=name)
        self.docking_status = docking_status

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        # status derived directly from simulation state, therefore no state machine needed
        pass

    def _process(self, sim_state):
        # process stare and return status
        success = sim_state.status[self.docking_status]
        return success
