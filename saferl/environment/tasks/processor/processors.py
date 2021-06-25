import abc
import numpy as np


class Processor(abc.ABC):
    def __init__(self, name=None):
        self.name = name

    @abc.abstractmethod
    def reset(self, sim_state):
        """Reset the processor instance"""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_info(self) -> dict:
        """Create and return an info dict"""
        raise NotImplementedError

    def step(self, sim_state, step_size):
        # two stage wrapper which encapsulates the transition between states of size 'step_size'
        self.increment(sim_state, step_size)
        return self.process(sim_state)

    def increment(self, sim_state, step_size):
        # method to expose the progression of internal status proportional to step size
        self._increment(sim_state, step_size)

    def process(self, sim_state):
        # method to expose internal state
        return self._process(sim_state)

    @abc.abstractmethod
    def _increment(self, sim_state, step_size):
        """
        A method to progress and update internal state proportional to the given step size.

        Parameters
        ----------
        sim_state : SimulationState
            simulation state member of parent environment
        step_size : float
            size of time increment
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _process(self, sim_state):
        """
        A method to process internal state and return relevant value(s).

        Parameters
        ----------
        sim_state : SimulationState
            simulation state member of parent environment
        Returns
        -------
        Processor-specific value (observations, reward value, status value, ect.) based off of current internal state.
        """
        raise NotImplementedError


class ObservationProcessor(Processor):
    def __init__(self, name=None, normalization=None, clipping_bounds=None):
        super().__init__(name=name)
        self.obs = None
        # self.observation_space = None

        self.normalization = normalization
        self.clipping_bounds = clipping_bounds

    def reset(self, sim_state):
        self.obs = None

    def generate_info(self) -> dict:
        info = {
            "observation": self.obs
        }
        return info

    def _normalize(self, obs):
        # apply normalization vector to given observations

        if self.normalization is None:
            # no normalization specified, so no change to observations
            return obs

        # ensure normalization vector is correct types
        assert type(self.normalization) in [np.array, np.ndarray], \
            "Expected numpy.array or numpy.ndarray for variable \'normalization\', but instead got: {}".format(
                type(self.normalization))
        # ensure normalization vector is correct shape
        assert obs.shape == self.normalization.shape, \
            "The shape of the observation space and normalization vector do not match!"

        # normalization vector is compatible, so return normalized observations
        obs = np.divide(obs, self.normalization)
        return obs

    def _clip(self, obs):
        # apply normalization vector to given observations

        if self.clipping_bounds is None:
            # no specified clipping
            return obs

        assert type(self.clipping_bounds) == dict, \
            "Expected dict for variable \'clipping_bounds\', but instead got: {}".format(type(self.clipping_bounds))
        assert "max" in self.clipping_bounds, "\'max\' key not defined in \'clipping_bounds\'"
        assert "min" in self.clipping_bounds, "\'min\' key not defined in \'clipping_bounds\'"

        # apply clipping in specified range
        obs = np.clip(obs, self.clipping_bounds["min"], self.clipping_bounds["max"])
        return obs

    def process(self, sim_state):
        """
        A method to expose the current normalized observation space.

        Parameters
        ----------
        sim_state : SimulationState
            The current state of the simulated environment.

        Returns
        -------
        obs : numpy.array, numpy.ndarray
            The agent's vectorized observation space.
        """
        # get observations from state
        obs = self._process(sim_state)
        # normalize observations
        obs = self._normalize(obs)
        # clip
        obs = self._clip(obs)
        return obs

    def _increment(self, sim_state, step_size):
        # observation processors will not have a state to update by default
        pass

    @abc.abstractmethod
    def _process(self, sim_state) -> np.ndarray:
        # process state and return relevant observation array
        raise NotImplementedError


class StatusProcessor(Processor):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.status_value = None

    @abc.abstractmethod
    def reset(self, sim_state):
        # reset internal state
        raise NotImplementedError

    def generate_info(self) -> dict:
        info = {
            "status": self.status_value
        }
        return info

    @abc.abstractmethod
    def _increment(self, sim_state, step_size):
        # update internal state over step transition
        raise NotImplementedError

    @abc.abstractmethod
    def _process(self, sim_state):
        # process state values to return status
        raise NotImplementedError


class RewardProcessor(Processor):
    def __init__(self, name=None, reward=None):
        super().__init__(name=name)
        self.step_value = 0
        self.total_value = 0
        self.reward = reward

    def reset(self, sim_state):
        self.step_value = 0
        self.total_value = 0

    def generate_info(self) -> dict:
        info = {
            "step": self.step_value,
            "total": self.total_value
        }
        return info

    def step(self, sim_state, step_size):
        # two stage wrapper which encapsulates the transition between states
        #   and computation of output
        self.increment(sim_state, step_size)
        self.step_value = self.process(sim_state)
        self.total_value += self.step_value
        return self.step_value

    @abc.abstractmethod
    def _increment(self, sim_state, step_size):
        # update internal state over step transition
        raise NotImplementedError

    @abc.abstractmethod
    def _process(self, sim_state):
        # calculate and return step value from current internal state
        raise NotImplementedError

    def get_step_value(self):
        return self.step_value

    def get_total_value(self):
        return self.total_value
