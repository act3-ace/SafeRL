import abc


class Processor(abc.ABC):
    def __init__(self, config):
        self.config = config["config"]
        self.name = config["name"]

    @abc.abstractmethod
    def reset(self, env_objs):
        """Reset the processor instance"""
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_info(self) -> dict:
        """Create and return an info dict"""
        raise NotImplementedError


    def step(self, env_objs, timestep, status):
        # two stage wrapper which encapsulates the transition between states of size 'timestep'
        self.increment(env_objs, timestep, status)
        return self.process(env_objs, status)

    @abc.abstractmethod
    def increment(self, env_objs, timestep, status):
        # method to progress internal state proportional to given timestep size
        raise NotImplementedError

    def process(self, env_objs, status):
        # method to process and return relevant status
        return self._process(env_objs, status)

    @abc.abstractmethod
    def _process(self, env_objs, status):
        raise NotImplementedError


class ObservationProcessor(Processor):
    def __init__(self, config):
        super().__init__(config=config)
        self.obs = None
        self.observation_space = None
        # TODO: add normalization and clipping, pre- and post- processors

    def reset(self, env_objs):
        self.obs = None

    def _generate_info(self) -> dict:
        info = {
            "observation": self.obs
        }
        return info


    def increment(self, env_objs, timestep, status):
        # generate observations
        self.obs = self._increment(env_objs, timestep, status)

    @abc.abstractmethod
    def _increment(self, env_objs, timestep, status):
        """Generate and return an observation from the current environment"""
        raise NotImplementedError

    # @abc.abstractmethod
    # def _process(self, env_objs, status):
    #     """Return processed observation from current environment"""
    #     raise NotImplementedError


class StatusProcessor(Processor):
    def __init__(self, config):
        super().__init__(config=config)
        self.status_value = None

    def reset(self, env_objs):
        self.status_value = None

    def _generate_info(self) -> dict:
        info = {
            "status": self.status_value
        }
        return info


    def increment(self, env_objs, timestep, status):
        self.status_value = self._increment(env_objs, timestep, status)

    @abc.abstractmethod
    def _increment(self, env_objs, timestep, status):
        """Generate a status value from the environment at the current timestep"""
        raise NotImplementedError

    # @abc.abstractmethod
    # def _process(self, env_objs, status):
    #     """return a processed status value from the environment at the current timestep"""
    #     raise NotImplementedError


class RewardProcessor(Processor):
    def __init__(self, config):
        super().__init__(config=config)
        self.step_value = 0
        self.total_value = 0
        self.reward = self.config["reward"]

    def reset(self, env_objs):
        self.step_value = 0
        self.total_value = 0

    def _generate_info(self) -> dict:
        info = {
            "step": self.step_value,
            "total": self.total_value
        }
        return info

    def increment(self, env_objs, timestep, status):
        self.step_value = self._increment(env_objs, timestep, status)
        self.total_value += self.step_value

    @abc.abstractmethod
    def _increment(self, env_objs, timestep, status):
        """Generate a reward from environment at current timestep"""
        raise NotImplementedError

    # @abc.abstractmethod
    # def _process(self, env_objs, status):
    #     """Return a processed reward from environment at current timestep"""
    #     raise NotImplementedError
