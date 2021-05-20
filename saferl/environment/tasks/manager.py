import abc
import copy
import numpy as np


class Manager(abc.ABC):
    def __init__(self, *processor_configs):
        # Register and initialize processors
        self.processors = [p_config["class"](config=p_config["config"]) for p_config in processor_configs]

    def reset(self, sim_state):
        for p in self.processors:
            p.reset(sim_state)

    @abc.abstractmethod
    def step(self, sim_state, step_size):
        """increment and process processors and return resulting value"""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_info(self) -> dict:
        """Create and return an info dict"""
        raise NotImplementedError

    @abc.abstractmethod
    def process(self, sim_state):
        """iteratively process processors and return resulting value"""
        raise NotImplementedError


class ObservationManager(Manager):
    def __init__(self, *processor_configs):
        super().__init__(processor_configs)
        self.obs = None

        # All processors should have same observation space
        self.observation_space = self.processors[0].observation_space

    def reset(self, sim_state):
        super().reset(sim_state)
        self.obs = None

    def generate_info(self) -> dict:
        info = {}
        return info

    def step(self, sim_state, step_size):
        obs_list = []
        for processor in self.processors:
            obs_list.append(processor.step(sim_state, step_size))
        self.obs = np.concatenate(obs_list)
        return self.obs

    def process(self, sim_state):
        obs_list = []
        for processor in self.processors:
            obs_list.append(processor.process(sim_state))
        obs = np.concatenate(obs_list)
        return obs


class StatusManager(Manager):
    def __init__(self, *processor_configs):
        super().__init__(processor_configs)
        self.status = {}

    def reset(self, sim_state):
        # construct new status from initial environment
        return self._compute_status(sim_state, reset=True)

    def step(self, sim_state, step_size):
        return self._compute_status(sim_state, step_size=step_size)

    def process(self, sim_state):
        return self._compute_status(sim_state)

    def generate_info(self) -> dict:
        info = {
            'status': self.status
        }
        return info

    def _compute_status(self, sim_state, step_size=None, reset=False):
        # construct new status within sim_state shallow copy
        self.status = {}
        sim_state_new = copy.copy(sim_state)
        sim_state_new.status = self.status

        for processor in self.processors:
            if reset:
                processor.reset(sim_state_new)

            if step_size is None:
                sim_state_new.status[processor.name] = processor.process(sim_state_new)
            else:
                sim_state_new.status[processor.name] = processor.step(sim_state_new, step_size)

        return sim_state_new.status


class RewardManager(Manager):
    def __init__(self, *processor_configs):
        super().__init__(processor_configs)
        self.step_value = 0
        self.total_value = 0

    def reset(self, sim_state):
        super().reset(sim_state)
        self.step_value = 0
        self.total_value = 0

    def generate_components(self):
        """helper method to organize reward components"""
        components = {"step": {}, "total": {}}
        for p in self.processors:
            components["step"][p.name] = p.get_step_value()
            components["total"][p.name] = p.get_total_value()
        return components

    def generate_info(self):
        info = {
            'step': self.step_value,
            'components': self.generate_components(),
            'total': self.total_value,
        }

        return info

    def step(self, sim_state, step_size):
        self.step_value = 0
        for processor in self.processors:
            self.step_value += processor.step(sim_state, step_size)
        self.total_value += self.step_value
        return self.step_value

    def process(self, sim_state):
        step_value = 0
        for processor in self.processors:
            step_value += processor.process(sim_state)
        return step_value
