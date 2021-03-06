import gym.spaces
import numpy as np

from saferl.aerospace.models.integrators.integrator_1d import Integrator1d
from saferl.aerospace.models.integrators.integrator_3d import Integrator3d
from saferl.aerospace.models.cwhspacecraft.platforms import CWHSpacecraft2d, CWHSpacecraft3d, CWHSpacecraftOriented2d
from saferl.environment.tasks.processor import ObservationProcessor, RewardProcessor, StatusProcessor
from saferl.environment.models.geometry import distance

# --------------------- Observation Processors ------------------------


class DockingObservationProcessor(ObservationProcessor):
    def __init__(self, name=None, deputy=None, mode='2d', normalization=None, clip=None, post_processors=None):
        # Initialize member variables from config
        # 2d or 3d
        self.mode = mode
        # not platform ref, string for name of deputy
        self.deputy = deputy

        # add default normalization
        if normalization is None:
            if self.mode == '2d':
                normalization = [100, 100, .5, .5, 1, 1]
            elif self.mode == '3d':
                normalization = [100, 100, 100, .5, .5, .5, 1, 1]

        # Invoke parent's constructor
        super().__init__(name=name, normalization=normalization, clip=clip, post_processors=post_processors)

    def define_observation_space(self) -> gym.spaces.Box:
        low = np.finfo(np.float32).min
        high = np.finfo(np.float32).max

        if self.mode == '2d':
            observation_space = gym.spaces.Box(low=low, high=high, shape=(6,))
        elif self.mode == '3d':
            observation_space = gym.spaces.Box(low=low, high=high, shape=(8,))
        else:
            raise ValueError("Invalid observation mode {}. Should be '1d', '2d', or '3d'.".format(self.mode))

        return observation_space

    def _process(self, sim_state):
        obs = np.copy(sim_state.env_objs[self.deputy].state.vector)
        obs = np.append(obs, np.linalg.norm(sim_state.env_objs[self.deputy].velocity))
        obs = np.append(obs, sim_state.status['max_vel_limit'])
        return obs


class DockingObservationProcessorOriented(ObservationProcessor):
    def __init__(self, name=None, deputy=None, mode='2d', normalization=None, clip=None, post_processors=None):
        # Initialize member variables from config
        self.mode = mode
        self.deputy = deputy

        # Invoke parent's constructor
        super().__init__(name=name, normalization=normalization, clip=clip, post_processors=post_processors)

        if not self.has_normalization:
            if self.mode == '2d':
                # if no custom normalization defined
                self._add_normalization([100, 100, np.pi, 0.5, 0.5, np.deg2rad(2), 1, 1])

    def define_observation_space(self) -> gym.spaces.Box:
        low = np.finfo(np.float32).min
        high = np.finfo(np.float32).max

        if self.mode == '2d':
            observation_space = gym.spaces.Box(low=low, high=high, shape=(8,))
        elif self.mode == '3d':
            raise NotImplementedError
        else:
            raise ValueError("Invalid observation mode {}. Should be '2d' or '3d'.".format(self.mode))

        return observation_space

    def _process(self, sim_state):
        obs = sim_state.env_objs[self.deputy].state.vector

        # if self.config['mode'] == '2d':
        #     obs[2]

        obs = np.append(obs, np.linalg.norm(sim_state.env_objs[self.deputy].velocity))
        obs = np.append(obs, sim_state.status['max_vel_limit'])

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
    def __init__(self, name=None, success_status=None, reward=None, timeout=None):
        super().__init__(name=name, reward=reward)
        self.success_status = success_status
        self.timeout = timeout

    def _increment(self, sim_state, step_size):
        # reward derived straight from status dict, therefore no state machine necessary
        pass

    def _process(self, sim_state):
        step_reward = 0
        if sim_state.status[self.success_status]:
            step_reward = self.reward
            if self.timeout is not None:
                step_reward += 1 - (sim_state.time_elapsed / self.timeout)
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


class InDockingStatusProcessor(StatusProcessor):
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


class DockingVelocityLimit(StatusProcessor):
    def __init__(self, name, target, dist_status, vel_threshold, threshold_dist, slope=2):
        self.target = target
        self.dist_status = dist_status
        self.vel_threshold = vel_threshold
        self.threshold_dist = threshold_dist
        self.slope = slope
        super().__init__(name)

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        pass

    def _process(self, sim_state):
        target_obj = sim_state.env_objs[self.target]
        dist = sim_state.status[self.dist_status]

        target_mean_motion = target_obj.dynamics.n

        vel_limit = self.vel_threshold

        if dist > self.threshold_dist:
            vel_limit += self.slope * target_mean_motion * (dist - self.threshold_dist)

        return vel_limit


class DockingVelocityLimitViolation(StatusProcessor):
    def __init__(self, name, target, ref, vel_limit_status, lower_bound=False):
        self.target = target
        self.ref = ref
        self.lower_bound = lower_bound
        self.vel_limit_status = vel_limit_status
        super().__init__(name)

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        pass

    def _process(self, sim_state):
        target_obj = sim_state.env_objs[self.target]
        ref_obj = sim_state.env_objs[self.ref]

        vel_limit = sim_state.status[self.vel_limit_status]

        rel_vel = target_obj.velocity - ref_obj.velocity
        rel_vel_mag = np.linalg.norm(rel_vel)

        violation = rel_vel_mag - vel_limit
        if self.lower_bound:
            violation *= -1

        return violation


class RelativeVelocityConstraint(StatusProcessor):
    def __init__(self, name, target, ref, vel_limit_status, lower_bound=False):
        self.target = target
        self.ref = ref
        self.lower_bound = lower_bound
        self.vel_limit_status = vel_limit_status
        super().__init__(name)

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        pass

    def _process(self, sim_state):
        target_obj = sim_state.env_objs[self.target]
        ref_obj = sim_state.env_objs[self.ref]

        vel_limit = sim_state.status[self.vel_limit_status]

        rel_vel = target_obj.velocity - ref_obj.velocity
        rel_vel_mag = np.linalg.norm(rel_vel)

        if self.lower_bound:
            return rel_vel_mag >= vel_limit
        else:
            return rel_vel_mag <= vel_limit


class SafetyConstraintsProcessor(StatusProcessor):
    def __init__(self, name, safety_constraint_statuses):
        self.safety_constraint_statuses = safety_constraint_statuses
        super().__init__(name)

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        # status derived directly from simulation state. No state machine necessary
        pass

    def _process(self, sim_state):
        in_docking = sim_state.env_objs[self.docking_region].contains(sim_state.env_objs[self.deputy])
        return in_docking


class DockingThrustDeltaVStatusProcessor(StatusProcessor):
    def __init__(self, name, target):
        super().__init__(name=name)
        self.target = target
        self.step_delta_v = 0

    def reset(self, sim_state):
        self.step_delta_v = 0

    def _increment(self, sim_state, step_size):
        # status derived directly from simulation state. No state machine necessary
        target_platform = sim_state.env_objs[self.target]
        assert isinstance(
            target_platform,
            (CWHSpacecraft2d, CWHSpacecraft3d, CWHSpacecraftOriented2d, Integrator1d, Integrator3d))
        control_vec = target_platform.current_control

        if isinstance(target_platform, CWHSpacecraftOriented2d):
            control_vec = control_vec[:-1]

        mass = target_platform.dynamics.m

        self.step_delta_v = np.sum(np.abs(control_vec)) / mass * step_size

    def _process(self, sim_state):
        return self.step_delta_v


class AccumulatorStatusProcessor(StatusProcessor):
    def __init__(self, name, status):
        super().__init__(name=name)
        self.status = status
        self.total = 0

    def reset(self, sim_state):
        self.total = 0

    def _increment(self, sim_state, step_size):
        # status derived directly from simulation state. No state machine necessary
        self.total += sim_state.status[self.status]

    def _process(self, sim_state):
        return self.total


class FailureStatusProcessor(StatusProcessor):
    def __init__(self,
                 name,
                 docking_distance,
                 max_goal_distance,
                 in_docking_status,
                 max_vel_constraint_status,
                 timeout):
        super().__init__(name=name)
        self.timeout = timeout
        self.docking_distance = docking_distance
        self.max_goal_distance = max_goal_distance
        self.in_docking_status = in_docking_status
        self.max_vel_constraint_status = max_vel_constraint_status

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
        elif sim_state.status[self.in_docking_status] and (not sim_state.status[self.max_vel_constraint_status]):
            failure = 'crash'
        else:
            failure = False
        return failure


class SuccessStatusProcessor(StatusProcessor):
    def __init__(self, name, in_docking_status, max_vel_constraint_status):
        super().__init__(name=name)
        self.in_docking_status = in_docking_status
        self.max_vel_constraint_status = max_vel_constraint_status

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        # status derived directly from simulation state, therefore no state machine needed
        pass

    def _process(self, sim_state):
        # process stare and return status
        success = sim_state.status[self.in_docking_status] and sim_state.status[self.max_vel_constraint_status]
        return success
