import abc


class Processor(abc.ABC):
    def __init__(self, config, name="generic_processor"):
        self.config = config
        self.name = name

    @abc.abstractmethod
    def reset(self, env_objs):
        """Reset the processor instance"""
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_info(self) -> dict:
        """Create and return an info dict"""
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, env_objs, timestep, status, old_status):
        """Process the environment at the current timestep"""
        raise NotImplementedError


class ObservationProcessor(Processor):
    def __init__(self, config, name="observation_processor"):
        super().__init__(config=config, name=name)
        self.obs = None
        self.observation_space = None
        # TODO: add normalization and clipping, pre- and post- processors

    def reset(self, env_objs):
        self.obs = None

    def step(self, env_objs, timestep, status, old_status):
        self.obs = self.generate_observation(env_objs=env_objs)

    def _generate_info(self) -> dict:
        info = {
            "observation": self.obs
        }
        return info

    @abc.abstractmethod
    def generate_observation(self, env_objs):
        """Generate an observation from the current environment"""
        raise NotImplementedError


class StatusProcessor(Processor):
    def __init__(self, config, name="status_processor"):
        super().__init__(config=config, name=name)
        self.status_value = None

    def reset(self, env_objs):
        self.status_value = None

    def _generate_info(self) -> dict:
        info = {
            "status": self.status_value
        }
        return info

    def step(self, env_objs, timestep, status, old_status):
        self.status_value = self.generate_status(
            env_objs=env_objs,
            timestep=timestep,
            status=status,
            old_status=old_status
        )
        status[self.name] = self.status_value

    @abc.abstractmethod
    def generate_status(self, env_objs, timestep, status, old_status):
        """Generate a status value from the environment at the current timestep"""
        raise NotImplementedError


class RewardProcessor(Processor):
    def __init__(self, config, name="reward_processor"):
        super().__init__(config=config, name=name)
        self.step_value = 0
        self.total_value = 0

    def reset(self, env_objs):
        self.step_value = 0
        self.total_value = 0

    def _generate_info(self) -> dict:
        info = {
            "step": self.step_value,
            "total": self.total_value
        }
        return info

    def step(self, env_objs, timestep, status, old_status=None):
        self.step_value = self.generate_reward(env_objs, timestep, status)
        self.total_value += self.step_value

    @abc.abstractmethod
    def generate_reward(self, env_objs, timestep, status):
        raise NotImplementedError
