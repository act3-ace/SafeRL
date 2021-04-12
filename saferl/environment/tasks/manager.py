import abc
import numpy as np


class Manager(abc.ABC):
    def __init__(self, config):
        self.config = config

        # Register and initialize processors
        self.processors = [p_config["class"](config=p_config) for p_config in config]

    def reset(self, env_objs):
        for p in self.processors:
            p.reset(env_objs)

    @abc.abstractmethod
    def step(self, env_objs, timestep, status):
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_info(self) -> dict:
        """Create and return an info dict"""
        raise NotImplementedError

    # @abc.abstractmethod
    # def process(self, env_objs, status):
    #     ...


class ObservationManager(Manager):
    def __init__(self, config):
        super().__init__(config=config)
        self.obs = None

        # All processors should have same observation space
        self.observation_space = self.processors[0].observation_space

    def _generate_info(self) -> dict:
        info = {}
        return info

    def step(self, env_objs, timestep, status):
        obs_list = []
        for processor in self.processors:
            obs_list.append(processor.step(env_objs, timestep, status))
        self.obs = np.concatenate(obs_list)
        return self.obs


class StatusManager(Manager):
    def __init__(self, config):
        super().__init__(config=config)
        self.status = {}

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.status = {}

    def _generate_info(self) -> dict:
        info = {
            'status': self.status
        }
        return info

    def step(self, env_objs, timestep, status):
        self.status = {}
        for processor in self.processors:
            self.status[processor.name] = processor.step(env_objs, timestep, self.status)

        return self.status


class RewardManager(Manager):
    def __init__(self, config):
        super().__init__(config=config)
        self.step_value = 0
        self.total_value = 0

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)

    def generate_components(self):
        """helper method to organize reward components"""
        components = {"step": {}, "total": {}}
        for p in self.processors:
            components["step"][p.name] = p.step_value
            components["total"][p.name] = p.total_value
        return components

    def _generate_info(self):
        info = {
            'step': self.step_value,
            'component_totals': self.generate_components(),
            'total': self.total_value,
        }

        return info

    def step(self, env_objs, timestep, status):
        self.step_value = 0
        for processor in self.processors:
            self.step_value += processor.step(env_objs, timestep, status)
        self.total_value += self.step_value
        return self.step_value
