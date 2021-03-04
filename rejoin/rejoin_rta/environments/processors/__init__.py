from abc import ABC, abstractmethod


class Processor(ABC):
    def __init__(self, config, name="generic_processor"):
        self.config = config
        self.step_value = 0
        self.total_value = 0
        self.name = name

    def reset(self, env_objs):
        """Reset the processor instance"""
        self.step_value = 0
        self.total_value = 0

    def _generate_info(self) -> dict:
        """Create and return an info dict"""
        info = {
            "step": self.step_value,
            "total": self.total_value
        }
        return info

    @abstractmethod
    def step(self, env_objs, time_step, status_dict):
        """Generate step reward"""
        raise NotImplementedError
