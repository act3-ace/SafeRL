import gym.spaces
import numpy as np
import math

from scipy.spatial.transform import Rotation

from saferl.environment.tasks import ObservationProcessor, RewardProcessor, StatusProcessor
from saferl.environment.models import distance


class DubinsObservationProcessor(ObservationProcessor):
    def __init__(self, config):
        super().__init__(config=config)

        # Initialize member variables from config
        self.lead = self.config["lead"]
        self.wingman = self.config["wingman"]
        self.rejoin_region = self.config["rejoin_region"]
        self.reference = self.config["reference"]
        self.mode = self.config["mode"]

        if self.mode == 'rect':
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(8,))
            self.obs_norm_const = np.array([10000, 10000, 10000, 10000, 100, 100, 100, 100], dtype=np.float64)
        elif self.mode == 'magnorm':
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(12,))
            self.obs_norm_const = np.array([10000, 1, 1, 10000, 1, 1, 100, 1, 1, 100, 1, 1], dtype=np.float64)

    def vec2magnorm(self, vec):
        norm = np.linalg.norm(vec)
        mag_norm_vec = np.concatenate(([norm], vec / norm))
        return mag_norm_vec

    def _process(self, env_objs, status):
        wingman_lead_r = env_objs[self.lead].position - env_objs[self.wingman].position
        wingman_rejoin_r = env_objs[self.rejoin_region].position - env_objs[self.wingman].position

        wingman_vel = env_objs[self.wingman].velocity
        lead_vel = env_objs[self.lead].velocity

        reference_rotation = Rotation.from_quat([0, 0, 0, 1])
        if self.reference == 'wingman':
            reference_rotation = env_objs[self.wingman].orientation.inv()

        wingman_lead_r = reference_rotation.apply(wingman_lead_r)
        wingman_rejoin_r = reference_rotation.apply(wingman_rejoin_r)

        wingman_vel = reference_rotation.apply(wingman_vel)
        lead_vel = reference_rotation.apply(lead_vel)

        if self.mode == 'magnorm':
            wingman_lead_r = self.vec2magnorm(wingman_lead_r)
            wingman_rejoin_r = self.vec2magnorm(wingman_rejoin_r)

            wingman_vel = self.vec2magnorm(wingman_vel)
            lead_vel = self.vec2magnorm(lead_vel)

        obs = np.concatenate([
            wingman_lead_r[0:3],
            wingman_rejoin_r[0:3],
            wingman_vel[0:3],
            lead_vel[0:3],
        ])

        # normalize observation
        obs = np.divide(obs, self.obs_norm_const)

        obs = np.clip(obs, -1, 1)

        return obs


class RejoinRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)

        # Initialize member variables from config
        self.rejoin_status = self.config["rejoin_status"]
        self.rejoin_prev_status = self.config["rejoin_prev_status"]

        # Initialize state vars
        self.timestep_size = 0
        self.in_rejoin_for_timestep = False
        self.left_rejoin = False

    def _increment(self, env_objs, timestep, status):
        # Update state variables
        self.left_rejoin = False
        self.in_rejoin_for_timestep = False
        self.timestep_size = timestep

        in_rejoin = status[self.rejoin_status]
        in_rejoin_prev = status[self.rejoin_prev_status]
        if in_rejoin and in_rejoin_prev:
            # determine if agent was in rejoin for duration of timestep
            self.in_rejoin_for_timestep = True
        elif not in_rejoin and in_rejoin_prev:
            # if rejoin region is left, refund all accumulated rejoin reward
            #   this is to ensure that the agent doesn't infinitely enter and leave rejoin region
            self.left_rejoin = True

    def _process(self, env_objs, status):
        # process state variables and return appropriate reward
        step_reward = 0
        if self.in_rejoin_for_timestep:
            step_reward = self.reward * self.timestep_size
        elif self.left_rejoin:
            step_reward = -1 * self.total_value
        return step_reward


class RejoinFirstTimeRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.rejoin_first_time_applied = False

        # Initialize member variables from config
        self.rejoin_status = self.config["rejoin_status"]

        # Initialize state vars
        self.rejoin_first_time = False
        self.rejoin_first_time_applied = False

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.rejoin_first_time = False
        self.rejoin_first_time_applied = False
        # TODO: handle init in rejoin case

    def _increment(self, env_objs, timestep, status):
        # return step_value of reward accumulated during interval of size timestep

        if self.rejoin_first_time:
            # if first time flag true, first time reward has already been applied
            self.rejoin_first_time_applied = True
            self.rejoin_first_time = False

        in_rejoin = status[self.rejoin_status]
        if in_rejoin and not self.rejoin_first_time_applied:
            # if in rejoin and first time reward not already applied, it is the agent's first timestep in rejoin region
            self.rejoin_first_time = True

    def _process(self, env_objs, status):
        # process state variables and return appropriate reward
        step_reward = 0
        if self.rejoin_first_time:
            step_reward = self.reward
        return step_reward


class RejoinDistanceChangeRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.prev_distance = 0
        # Initialize member variables from config
        self.rejoin_status = self.config["rejoin_status"]
        self.wingman = self.config["wingman"]
        self.rejoin_region = self.config["rejoin_region"]

        # Initialize state vars
        self.prev_distance = 0
        self.cur_distance = 0
        self.in_rejoin = False

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.prev_distance = distance(env_objs[self.wingman], env_objs[self.rejoin_region])
        self.cur_distance = self.prev_distance
        # TODO: handle init in rejoin case

    def _increment(self, env_objs, timestep, status):
        # Update state variables
        self.prev_distance = self.cur_distance
        self.cur_distance = distance(env_objs[self.wingman], env_objs[self.rejoin_region])
        self.in_rejoin = status[self.rejoin_status]

    def _process(self, env_objs, status):
        # process state and return appropriate step reward
        step_reward = 0
        distance_change = self.cur_distance - self.prev_distance
        if not self.in_rejoin:
            step_reward = distance_change * self.reward
        return step_reward


class DubinsInRejoin(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        # Initialize member variables from config
        self.wingman = self.config["wingman"]
        self.rejoin_region = self.config["rejoin_region"]

    def generate_status(self, env_objs):
        in_rejoin = env_objs[self.rejoin_region].contains(env_objs[self.wingman])
        return in_rejoin

    def reset(self, env_objs, status):
        return self.generate_status(env_objs)

    def _increment(self, env_objs, timestep, status):
        # this status comes from first principles of the environment and therefore does not require a state machine
        ...

    def _process(self, env_objs, status):
        # return the current status
        return self.generate_status(env_objs)


class DubinsInRejoinPrev(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        # Initialize member variables from config
        self.rejoin_status = self.config["rejoin_status"]

        # Initialize state vars
        self.in_rejoin_prev = False
        self.in_rejoin_current = False

    def reset(self, env_objs, status):
        self.in_rejoin_prev = False
        self.in_rejoin_current = status[self.rejoin_status]
        return self.in_rejoin_prev

    def _increment(self, env_objs, timestep, status):
        # update rejoin region state variables
        self.in_rejoin_prev = self.in_rejoin_current
        self.in_rejoin_current = status[self.rejoin_status]

    def _process(self, env_objs, status):
        # return the previous rejoin status
        return self.in_rejoin_prev


class DubinsRejoinTime(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.rejoin_time = 0
        # Initialize member variables from config
        self.rejoin_status = self.config["rejoin_status"]

        # Initialize state vars
        self.rejoin_time = 0
        self.in_rejoin = False

    def reset(self, env_objs, status):
        self.rejoin_time = 0
        self.in_rejoin = status[self.rejoin_status]
        return self.rejoin_time

    def _increment(self, env_objs, timestep, status):
        # update state
        self.in_rejoin = status[self.rejoin_status]
        if self.in_rejoin:
            self.rejoin_time += timestep
        else:
            self.rejoin_time = 0

    def _process(self, env_objs, status):
        # return the current status
        return self.rejoin_time


class DubinsTimeElapsed(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)

        # Initialize state var
        self.time_elapsed = 0

    def reset(self, env_objs, status):
        self.time_elapsed = 0
        return self.time_elapsed

    def _increment(self, env_objs, timestep, status):
        # update status
        self.time_elapsed += timestep

    def _process(self, env_objs, status):
        # return the current status
        return self.time_elapsed


class DubinsLeadDistance(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        # Initialize member variables from config
        self.wingman = self.config["wingman"]
        self.lead = self.config["lead"]

    def generate_status(self, env_objs):
        lead_distance = distance(env_objs[self.wingman], env_objs[self.lead])
        return lead_distance

    def reset(self, env_objs, status):
        return self.generate_status(env_objs)

    def _increment(self, env_objs, timestep, status):
        # this status comes from first principles of the environment and therefore does not require a state machine
        ...

    def _process(self, env_objs, status):
        # return the current status
        return self.generate_status(env_objs)


class DubinsFailureStatus(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        # Initialize member variables from config
        self.lead_distance_key = self.config["lead_distance"]
        self.time_elapsed_key = self.config["time_elapsed"]
        self.safety_margin = self.config['safety_margin']
        self.timeout = self.config['timeout']
        self.max_goal_dist = self.config['max_goal_distance']

        # Initialize state vars
        self.lead_distance = 0
        self.time_elapsed = 0

    def generate_status(self):
        # process current state variables and return failure status
        failure = False
        if self.lead_distance < self.safety_margin['aircraft']:
            failure = 'crash'
        elif self.time_elapsed > self.timeout:
            failure = 'timeout'
        elif self.lead_distance >= self.max_goal_dist:
            failure = 'distance'

        return failure

    def reset(self, env_objs, status):
        # reset state and return status
        self.lead_distance = status[self.lead_distance_key]
        self.time_elapsed = status[self.time_elapsed_key]
        return self.generate_status()

    def _increment(self, env_objs, timestep, status):
        # update state
        self.lead_distance = status[self.lead_distance_key]
        self.time_elapsed = status[self.time_elapsed_key]

    def _process(self, env_objs, status):
        # process state and return status
        return self.generate_status()


class DubinsSuccessStatus(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        # Initialize member variables from config
        self.rejoin_time_key = self.config["rejoin_time"]
        self.success_time = self.config["success_time"]

        # Initialize state var
        self.rejoin_time = 0

    def generate_status(self):
        success = False

        if self.rejoin_time > self.success_time:
            success = True

        return success

    def reset(self, env_objs, status):
        # reset state and process new status
        self.rejoin_time = status[self.rejoin_time_key]
        return self.generate_status()

    def _increment(self, env_objs, timestep, status):
        # update state
        self.rejoin_time = status[self.rejoin_time_key]

    def _process(self, env_objs, status):
        # return new status
        return self.generate_status()
