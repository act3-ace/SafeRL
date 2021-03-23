import gym.spaces
import numpy as np

from scipy.spatial.transform import Rotation

from aerospaceSafeRL.environment.tasks import ObservationProcessor, RewardProcessor, StatusProcessor
from aerospaceSafeRL.environment.models import distance


class DubinsObservationProcessor(ObservationProcessor):
    def __init__(self, config):
        super().__init__(config=config, name="dubins_observation")

        if self.config['mode'] == 'rect':
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(8,))
            self.obs_norm_const = np.array([10000, 10000, 10000, 10000, 100, 100, 100, 100], dtype=np.float64)

        elif self.config['mode'] == 'magnorm':
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(12,))
            self.obs_norm_const = np.array([10000, 1, 1, 10000, 1, 1, 100, 1, 1, 100, 1, 1], dtype=np.float64)

    def generate_observation(self, env_objs):
        def vec2magnorm(vec):
            norm = np.linalg.norm(vec)
            mag_norm_vec = np.concatenate(([norm], vec / norm))
            return mag_norm_vec

        wingman_lead_r = env_objs['lead'].position - env_objs['wingman'].position
        wingman_rejoin_r = env_objs['rejoin_region'].position - env_objs['wingman'].position

        wingman_vel = env_objs['wingman'].velocity
        lead_vel = env_objs['lead'].velocity

        reference_rotation = Rotation.from_quat([0, 0, 0, 1])
        if self.config['reference'] == 'wingman':
            reference_rotation = env_objs['wingman'].orientation.inv()

        wingman_lead_r = reference_rotation.apply(wingman_lead_r)
        wingman_rejoin_r = reference_rotation.apply(wingman_rejoin_r)

        wingman_vel = reference_rotation.apply(wingman_vel)
        lead_vel = reference_rotation.apply(lead_vel)

        if self.config['mode'] == 'magnorm':
            wingman_lead_r = vec2magnorm(wingman_lead_r)
            wingman_rejoin_r = vec2magnorm(wingman_rejoin_r)

            wingman_vel = vec2magnorm(wingman_vel)
            lead_vel = vec2magnorm(lead_vel)

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
    def __init__(self, config, name="rejoin"):
        super().__init__(config=config, name=name)

    def generate_reward(self, env_objs, timestep, status):
        step_reward = 0
        in_rejoin = status["rejoin_status"]
        if in_rejoin:
            step_reward += self.config['rejoin_timestep'] * timestep
        else:
            # if rejoin region is left, refund all accumulated rejoin reward
            #   this is to ensure that the agent doesn't infinitely enter and leave rejoin region
            in_rejoin_prev = status["rejoin_prev_status"]
            if in_rejoin_prev:
                step_reward += -1 * self.total_value
        return step_reward


class RejoinFirstTimeRewardProcessor(RewardProcessor):
    def __init__(self, config, name="rejoin_first_time"):
        super().__init__(config=config, name=name)
        self.rejoin_first_time_applied = False

    def reset(self, env_objs):
        self.rejoin_first_time_applied = False

    def generate_reward(self, env_objs, timestep, status):
        step_reward = 0
        in_rejoin = status["rejoin_status"]
        if in_rejoin and not self.rejoin_first_time_applied:
            step_reward += self.config['rejoin_first_time']
            self.rejoin_first_time_applied = True
        return step_reward


class RejoinDistanceChangeRewardProcessor(RewardProcessor):
    def __init__(self, config, name="rejoin_distance"):
        super().__init__(config=config, name=name)
        self.prev_distance = 0

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.prev_distance = distance(env_objs['wingman'], env_objs['rejoin_region'])

    def generate_reward(self, env_objs, timestep, status):
        cur_distance = distance(env_objs['wingman'], env_objs['rejoin_region'])
        dist_change = cur_distance - self.prev_distance
        self.prev_distance = cur_distance

        in_rejoin = status["rejoin_status"]
        step_reward = 0
        if not in_rejoin:
            step_reward = dist_change * self.config['dist_change']
        return step_reward


class DubinsInRejoin(StatusProcessor):
    def __init__(self, config, name="rejoin_status"):
        super().__init__(config=config, name=name)

    def generate_status(self, env_objs, timestep, status, old_status):
        in_rejoin = env_objs['rejoin_region'].contains(env_objs['wingman'])
        return in_rejoin


class DubinsInRejoinPrev(StatusProcessor):
    def __init__(self, config, name="rejoin_prev_status"):
        super().__init__(config=config, name=name)

    def generate_status(self, env_objs, timestep, status, old_status):
        in_rejoin_prev = False
        if old_status:
            in_rejoin_prev = old_status["rejoin_status"]
        return in_rejoin_prev


class DubinsRejoinTime(StatusProcessor):
    def __init__(self, config, name="rejoin_time"):
        super().__init__(config=config, name=name)
        self.rejoin_time = 0

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.rejoin_time = 0

    def generate_status(self, env_objs, timestep, status, old_status):
        if status["rejoin_status"]:
            self.rejoin_time += timestep
        else:
            self.rejoin_time = 0
        return self.rejoin_time


class DubinsTimeElapsed(StatusProcessor):
    def __init__(self, config, name="rejoin_time_elapsed"):
        super().__init__(config=config, name=name)
        self.time_elapsed = 0

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.time_elapsed = 0

    def generate_status(self, env_objs, timestep, status, old_status):
        self.time_elapsed += timestep
        return self.time_elapsed


class DubinsLeadDistance(StatusProcessor):
    def __init__(self, config, name="rejoin_lead_distance"):
        super().__init__(config=config, name=name)

    def generate_status(self, env_objs, timestep, status, old_status):
        lead_distance = distance(env_objs['wingman'], env_objs['lead'])
        return lead_distance


class DubinsFailureStatus(StatusProcessor):
    def __init__(self, config, name="failure"):
        super().__init__(config=config, name=name)

    def generate_status(self, env_objs, timestep, status, old_status):
        lead_distance = status["rejoin_lead_distance"]
        time_elapsed = status["rejoin_time_elapsed"]

        failure = False

        if lead_distance < self.config['safety_margin']['aircraft']:
            failure = 'crash'
        elif time_elapsed > self.config['timeout']:
            failure = 'timeout'
        elif lead_distance >= self.config['max_goal_distance']:
            failure = 'distance'

        return failure


class DubinsSuccessStatus(StatusProcessor):
    def __init__(self, config, name="success"):
        super().__init__(config=config, name=name)

    def generate_status(self, env_objs, timestep, status, old_status):
        rejoin_time = status["rejoin_time"]

        success = False

        if rejoin_time > self.config['success']['rejoin_time']:
            success = True

        return success
