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

    def increment(self, env_objs, timestep, status):
        # method to expose the progression of internal status proportional to timestep size
        self._increment(env_objs, timestep, status)

    def process(self, env_objs, status):
        # method to expose internal state
        return self._process(env_objs, status)

    @abc.abstractmethod
    def _increment(self, env_objs, timestep, status):
        # method to progress internal state proportional to given timestep size
        raise NotImplementedError

    @abc.abstractmethod
    def _process(self, env_objs, status):
        # method to process and return relevant state
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

    def _increment(self, env_objs, timestep, status):
        """observation processors will not have a state to update by default"""
        ...

    @abc.abstractmethod
    def _process(self, env_objs, status):
        """method to process and return relevant state"""
        raise NotImplementedError


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

    @abc.abstractmethod
    def _increment(self, env_objs, timestep, status):
        """update state values from environment at the current timestep"""
        raise NotImplementedError

    @abc.abstractmethod
    def _process(self, env_objs, status):
        """method to process and return relevant state"""
        raise NotImplementedError


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


    def step(self, env_objs, timestep, status):
        # two stage wrapper which encapsulates the transition between states of size 'timestep'
        self.increment(env_objs, timestep, status)
        self.step_value = self.process(env_objs, status)
        self.total_value += self.step_value
        return self.step_value

    @abc.abstractmethod
    def _increment(self, env_objs, timestep, status):
        """update state from environment at current timestep"""
        raise NotImplementedError

    @abc.abstractmethod
    def _process(self, env_objs, status):
        """calculate and return step value from current internal state"""
        raise NotImplementedError
