from rejoin.rejoin_rta.utils.geometry import distance
from rejoin.rejoin_rta.environments.processors import RewardProcessor


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
