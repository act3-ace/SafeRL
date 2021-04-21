import gym.spaces
import numpy as np
import math

from scipy.spatial.transform import Rotation

from saferl.environment.tasks.processor import ObservationProcessor, RewardProcessor, StatusProcessor
from saferl.environment.models.geometry import distance


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

    def generate_observation(self, env_objs):
        def vec2magnorm(vec):
            norm = np.linalg.norm(vec)
            mag_norm_vec = np.concatenate(([norm], vec / norm))
            return mag_norm_vec

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
            wingman_lead_r = vec2magnorm(wingman_lead_r)
            wingman_rejoin_r = vec2magnorm(wingman_rejoin_r)

            wingman_vel = vec2magnorm(wingman_vel)
            lead_vel = vec2magnorm(lead_vel)

        obs = np.concatenate([
            wingman_lead_r,
            wingman_rejoin_r,
            wingman_vel,
            lead_vel
        ])

        # normalize observation
        obs = np.divide(obs, self.obs_norm_const)

        obs = np.clip(obs, -1, 1)

        return obs


class Dubins3dObservationProcessor(ObservationProcessor):
    def __init__(self, config):
        super().__init__(config=config)

        # Initialize member variables from config
        self.lead = self.config["lead"]
        self.wingman = self.config["wingman"]
        self.rejoin_region = self.config["rejoin_region"]
        self.reference = self.config["reference"]
        self.mode = self.config["mode"]

        if self.mode == 'rect':
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(14,))
            self.obs_norm_const = np.array([10000, 10000, 10000, 10000, 10000, 10000, 100, 100, 100, 100, 100, 100, math.pi, math.pi], dtype=np.float64)
        elif self.mode == 'magnorm':
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(18,))
            self.obs_norm_const = np.array([10000, 1, 1, 1, 10000, 1, 1, 1, 100, 1, 1, 1, 100, 1, 1, 1, math.pi, math.pi], dtype=np.float64)

    def generate_observation(self, env_objs):
        def vec2magnorm(vec):
            norm = np.linalg.norm(vec)
            mag_norm_vec = np.concatenate(([norm], vec / norm))
            return mag_norm_vec

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
            wingman_lead_r = vec2magnorm(wingman_lead_r)
            wingman_rejoin_r = vec2magnorm(wingman_rejoin_r)

            wingman_vel = vec2magnorm(wingman_vel)
            lead_vel = vec2magnorm(lead_vel)

        # gamma and roll for 3d orientation info
        roll = np.array([env_objs["wingman"].roll], dtype=np.float64)
        gamma = np.array([env_objs["wingman"].gamma], dtype=np.float64)

        obs = np.concatenate([
            wingman_lead_r,
            wingman_rejoin_r,
            wingman_vel,
            lead_vel,
            roll,
            gamma
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

    def generate_reward(self, env_objs, timestep, status):
        step_reward = 0
        in_rejoin = status[self.rejoin_status]
        if in_rejoin:
            step_reward += self.reward * timestep
        else:
            # if rejoin region is left, refund all accumulated rejoin reward
            #   this is to ensure that the agent doesn't infinitely enter and leave rejoin region
            in_rejoin_prev = status[self.rejoin_prev_status]
            if in_rejoin_prev:
                step_reward += -1 * self.total_value
        return step_reward


class RejoinFirstTimeRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.rejoin_first_time_applied = False

        # Initialize member variables from config
        self.rejoin_status = self.config["rejoin_status"]

    def reset(self, env_objs):
        self.rejoin_first_time_applied = False

    def generate_reward(self, env_objs, timestep, status):
        step_reward = 0
        in_rejoin = status[self.rejoin_status]
        if in_rejoin and not self.rejoin_first_time_applied:
            step_reward += self.reward
            self.rejoin_first_time_applied = True
        return step_reward


class RejoinDistanceChangeRewardProcessor(RewardProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.prev_distance = 0
        # Initialize member variables from config
        self.rejoin_status = self.config["rejoin_status"]
        self.wingman = self.config["wingman"]
        self.rejoin_region = self.config["rejoin_region"]

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.prev_distance = distance(env_objs[self.wingman], env_objs[self.rejoin_region])

    def generate_reward(self, env_objs, timestep, status):
        cur_distance = distance(env_objs[self.wingman], env_objs[self.rejoin_region])
        dist_change = cur_distance - self.prev_distance
        self.prev_distance = cur_distance

        in_rejoin = status[self.rejoin_status]
        step_reward = 0
        if not in_rejoin:
            step_reward = dist_change * self.reward
        return step_reward


class DubinsInRejoin(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        # Initialize member variables from config
        self.wingman = self.config["wingman"]
        self.rejoin_region = self.config["rejoin_region"]

    def generate_status(self, env_objs, timestep, status, old_status):
        in_rejoin = env_objs[self.rejoin_region].contains(env_objs[self.wingman])
        return in_rejoin


class DubinsInRejoinPrev(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        # Initialize member variables from config
        self.rejoin_status = self.config["rejoin_status"]

    def generate_status(self, env_objs, timestep, status, old_status):
        in_rejoin_prev = False
        if old_status:
            in_rejoin_prev = old_status[self.rejoin_status]
        return in_rejoin_prev


class DubinsRejoinTime(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.rejoin_time = 0
        # Initialize member variables from config
        self.rejoin_status = self.config["rejoin_status"]

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.rejoin_time = 0

    def generate_status(self, env_objs, timestep, status, old_status):
        if status[self.rejoin_status]:
            self.rejoin_time += timestep
        else:
            self.rejoin_time = 0
        return self.rejoin_time


class DubinsTimeElapsed(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        self.time_elapsed = 0

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.time_elapsed = 0

    def generate_status(self, env_objs, timestep, status, old_status):
        self.time_elapsed += timestep
        return self.time_elapsed


class DubinsLeadDistance(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        # Initialize member variables from config
        self.wingman = self.config["wingman"]
        self.lead = self.config["lead"]

    def generate_status(self, env_objs, timestep, status, old_status):
        lead_distance = distance(env_objs[self.wingman], env_objs[self.lead])
        return lead_distance


class DubinsFailureStatus(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        # Initialize member variables from config
        self.lead_distance = self.config["lead_distance"]
        self.time_elapsed = self.config["time_elapsed"]
        self.safety_margin = self.config['safety_margin']
        self.timeout = self.config['timeout']
        self.max_goal_dist = self.config['max_goal_distance']

    def generate_status(self, env_objs, timestep, status, old_status):
        lead_distance = status[self.lead_distance]
        time_elapsed = status[self.time_elapsed]

        failure = False

        if lead_distance < self.safety_margin['aircraft']:
            failure = 'crash'
        elif time_elapsed > self.timeout:
            failure = 'timeout'
        elif lead_distance >= self.max_goal_dist:
            failure = 'distance'

        return failure


class DubinsSuccessStatus(StatusProcessor):
    def __init__(self, config):
        super().__init__(config=config)
        # Initialize member variables from config
        self.rejoin_time = self.config["rejoin_time"]
        self.success_time = self.config["success_time"]

    def generate_status(self, env_objs, timestep, status, old_status):
        rejoin_time = status[self.rejoin_time]

        success = False

        if rejoin_time > self.success_time:
            success = True

        return success
