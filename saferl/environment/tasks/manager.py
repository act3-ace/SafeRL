import abc
import numpy as np


class Manager(abc.ABC):
    def __init__(self, config):
        self.config = config

        # Register and initialize processors
        self.processors = [p_config["class"](config=p_config) for p_config in config]

    def reset(self, env_objs, status):
        for p in self.processors:
            p.reset(env_objs, status)

    @abc.abstractmethod
    def step(self, env_objs, step_size, status):
        """increment and process processors and return resulting value"""
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_info(self) -> dict:
        """Create and return an info dict"""
        raise NotImplementedError

    @abc.abstractmethod
    def process(self, env_objs, status):
        """iteratively process processors and return resulting value"""
        raise NotImplementedError


class ObservationManager(Manager):
    def __init__(self, config):
        super().__init__(config=config)
        self.obs = None

        # All processors should have same observation space
        self.observation_space = self.processors[0].observation_space

    def reset(self, env_objs, status):
        super().reset(env_objs, status)
        self.obs = None
        
    def _generate_info(self) -> dict:
        info = {}
        return info

    def step(self, env_objs, step_size, status):
        obs_list = []
        for processor in self.processors:
            obs_list.append(processor.step(env_objs, step_size, status))
        self.obs = np.concatenate(obs_list)
        return self.obs

    def process(self, env_objs, status):
        obs_list = []
        for processor in self.processors:
            obs_list.append(processor.process(env_objs, status))
        obs = np.concatenate(obs_list)
        return obs


class StatusManager(Manager):
    def __init__(self, config):
        super().__init__(config=config)
        self.status = {}

    def reset(self, env_objs, status):
        # construct new status from initial environment
        self.status = {}
        for p in self.processors:
            p.reset(env_objs, self.status)
            self.status[p.name] = p.process(env_objs, self.status)
        return self.status

    def _generate_info(self) -> dict:
        info = {
            'status': self.status
        }
        return info

    def step(self, env_objs, step_size, status):
        self.status = {}
        for processor in self.processors:
            self.status[processor.name] = processor.step(env_objs, step_size, self.status)
        return self.status

    def process(self, env_objs, status):
        status = {}
        for processor in self.processors:
            status[processor.name] = processor.process(env_objs, status)
        return status


class RewardManager(Manager):
    def __init__(self, config):
        super().__init__(config=config)
        self.step_value = 0
        self.total_value = 0

    def reset(self, env_objs, status):
        super().reset(env_objs, status)
        self.step_value = 0
        self.total_value = 0

    def generate_components(self):
        """helper method to organize reward components"""
        components = {"step": {}, "total": {}}
        for p in self.processors:
            components["step"][p.name] = p.get_step_value()
            components["total"][p.name] = p.get_total_value()
        return components

    def _generate_info(self):
        info = {
            'step': self.step_value,
            'components': self.generate_components(),
            'total': self.total_value,
        }

        return info

    def step(self, env_objs, step_size, status):
        self.step_value = 0
        for processor in self.processors:
            self.step_value += processor.step(env_objs, step_size, status)
        self.total_value += self.step_value
        return self.step_value

    def process(self, env_objs, status):
        step_value = 0
        for processor in self.processors:
            step_value += processor.process(env_objs, status)
        return step_value
