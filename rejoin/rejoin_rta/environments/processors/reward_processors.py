from rejoin.rejoin_rta.utils.geometry import distance

from rejoin.rejoin_rta.environments.processors import Processor


class TimeRewardProcessor(Processor):
    def __init__(self, config, name="time"):
        super().__init__(config=config, name=name)

    def step(self, env_objs, time_step, status_dict):
        self.step_value = self.config["time_decay"]
        self.total_value += self.step_value
        return self.step_value


class DistanceChangeRewardProcessor(Processor):
    def __init__(self, config, name="distance"):
        super().__init__(config=config, name=name)
        self.prev_distance = 0

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.prev_distance = distance(env_objs['deputy'], env_objs['docking_region'])

    def step(self, env_objs, time_step, status_dict):
        cur_distance = distance(env_objs['deputy'], env_objs['docking_region'])
        dist_change = cur_distance - self.prev_distance
        self.prev_distance = cur_distance
        self.step_value = dist_change * self.config['dist_change']
        self.total_value += self.step_value
        return self.step_value


class DistanceChangeZRewardProcessor(Processor):
    def __init__(self, config, name="distance_z"):
        super().__init__(config=config, name=name)
        self.prev_distance = 0

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.prev_distance = abs(env_objs['deputy'].z - env_objs['docking_region'].z)

    def step(self, env_objs, time_step, status_dict):
        cur_distance_z = abs(env_objs['deputy'].z - env_objs['docking_region'].z)
        dist_z_change = cur_distance_z - self.prev_distance
        self.prev_distance = cur_distance_z
        self.step_value += dist_z_change * self.config['dist_z_change']
        self.total_value += self.step_value
        return self.step_value


class SuccessRewardProcessor(Processor):
    def __init__(self, config, name="success"):
        super().__init__(config=config, name=name)

    def step(self, env_objs, time_step, status_dict):
        if status_dict["success"]:
            self.step_value = self.config["success"]
        self.total_value += self.step_value
        return self.step_value


class FailureRewardProcessor(Processor):
    def __init__(self, config, name="failure"):
        super().__init__(config=config, name=name)

    def step(self, env_objs, time_step, status_dict):
        if status_dict["failure"]:
            self.step_value = self.config["failure"][status_dict['failure']]
        self.total_value += self.step_value
        return self.step_value
