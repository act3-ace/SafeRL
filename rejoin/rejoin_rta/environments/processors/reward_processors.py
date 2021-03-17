from rejoin.rejoin_rta.utils.geometry import distance
from rejoin.rejoin_rta.environments.processors import RewardProcessor


# TODO: Implement Dubins reward processors


class TimeRewardProcessor(RewardProcessor):
    def __init__(self, config, name="time"):
        super().__init__(config=config, name=name)

    def generate_reward(self, env_objs, timestep, status):
        step_reward = self.config["time_decay"]
        return step_reward


class DistanceChangeRewardProcessor(RewardProcessor):
    def __init__(self, config, name="distance"):
        super().__init__(config=config, name=name)
        self.prev_distance = 0

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.prev_distance = distance(env_objs['deputy'], env_objs['docking_region'])

    def generate_reward(self, env_objs, timestep, status):
        cur_distance = distance(env_objs['deputy'], env_objs['docking_region'])
        dist_change = cur_distance - self.prev_distance
        self.prev_distance = cur_distance
        step_reward = dist_change * self.config['dist_change']
        return step_reward


class DistanceChangeZRewardProcessor(RewardProcessor):
    def __init__(self, config, name="distance_z"):
        super().__init__(config=config, name=name)
        self.prev_distance = 0

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.prev_distance = abs(env_objs['deputy'].z - env_objs['docking_region'].z)

    def generate_reward(self, env_objs, timestep, status):
        cur_distance_z = abs(env_objs['deputy'].z - env_objs['docking_region'].z)
        dist_z_change = cur_distance_z - self.prev_distance
        self.prev_distance = cur_distance_z
        step_reward = dist_z_change * self.config['dist_z_change']
        return step_reward


class SuccessRewardProcessor(RewardProcessor):
    def __init__(self, config, name="success"):
        super().__init__(config=config, name=name)

    def generate_reward(self, env_objs, timestep, status):
        step_reward = 0
        if status["success"]:
            step_reward = self.config["success"]
        return step_reward


class FailureRewardProcessor(RewardProcessor):
    def __init__(self, config, name="failure"):
        super().__init__(config=config, name=name)

    def generate_reward(self, env_objs, timestep, status):
        step_reward = 0
        if status["failure"]:
            step_reward = self.config["failure"][status['failure']]
        return step_reward


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
