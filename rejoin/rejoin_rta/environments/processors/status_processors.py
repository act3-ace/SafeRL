from rejoin.rejoin_rta.utils.geometry import distance
from rejoin.rejoin_rta.environments.processors import StatusProcessor


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
