from rejoin.rejoin_rta.utils.geometry import distance
from rejoin.rejoin_rta.environments.processors import StatusProcessor


# TODO: Implement Dubins status processors


class DockingStatusProcessor(StatusProcessor):
    def __init__(self, config, name="docking_status"):
        super().__init__(config=config, name=name)

    def generate_status(self, env_objs, timestep, status, old_status):
        in_docking = env_objs['docking_region'].contains(env_objs['deputy'])
        return in_docking


class DockingDistanceStatusProcessor(StatusProcessor):
    def __init__(self, config, name="docking_distance"):
        super().__init__(config=config, name=name)

    def generate_status(self, env_objs, timestep, status, old_status):
        docking_distance = distance(env_objs['deputy'], env_objs['docking_region'])
        return docking_distance


class FailureStatusProcessor(StatusProcessor):
    def __init__(self, config, name="failure"):
        super().__init__(config=config, name=name)
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
    def __init__(self, config, name="success"):
        super().__init__(config=config, name=name)

    def generate_status(self, env_objs, timestep, status, old_status):
        success = status["docking_status"]
        return success
