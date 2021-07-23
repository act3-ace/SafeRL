import abc
import numpy as np
from collections.abc import Iterable

from saferl.environment.utils import Normalize, Clip


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
    def __init__(self, name=None, normalization=None, clip=None, post_processors=None):
        """
        The class constructor handles the assignment of member variables.

        Parameters
        ----------
        name : str
            The name of the processor to be displayed in logging.
        normalization : list or numpy.ndarray
            An array of constants used to normalize the values in a generated observation arrays via element-wise
            division.
        clip : list
            A two element list containing a minimum value boundary and a maximum value boundary (in that order) applied
            to all values in generated observation arrays.
        post_processors : list
            A list of dicts, each with PostProcessor class and a config dict KVPs.
        """

        super().__init__(name=name)
        self.obs = None
        # self.observation_space = None

        self.normalization = np.array(normalization, dtype=np.float64) if type(normalization) is list else normalization
        self.clip = clip                            # clip[0] == min clip bound, clip[1] == max clip bound
        self.post_processors = []                   # list of PostProcessors

        # create and store post processors
        if isinstance(post_processors, Iterable):
            for post_processor in post_processors:
                assert "class" in post_processor, \
                    "No 'class' key found in {} for construction of PostProcessor.".format(post_processor)
                assert "config" in post_processor, \
                    "No 'config' key found in {} for construction of PostProcessor.".format(post_processor)
                self.post_processors.append(post_processor["class"](**post_processor["config"]))

        # add norm + clipping postprocessors
        if self.normalization is not None:
            self._add_normalization(self.normalization)
        if self.clip is not None:
            self._add_clipping(self.clip)

    def reset(self, sim_state):
        self.obs = None

    def generate_info(self) -> dict:
        info = {
            "observation": self.obs
        }
        return info

    def _add_normalization(self, normalization_vector):
        # create normalization PostProcessor and add it to list
        # TODO: directly support setting mu
        normalization_post_proc = Normalize(sigma=normalization_vector)
        self.post_processors.append(normalization_post_proc)

    def _add_clipping(self, clip_bounds):
        # ensure clip_bounds format
        assert type(clip_bounds) == list, \
            "Expected a list for variable \'clip\', but instead got: {}".format(type(clip_bounds))
        assert 2 == len(clip_bounds), \
            "Expected a list of length 2 for variable \'clip\', but instead got: {}".format(len(clip_bounds))

        # create clipping PostProcessor and add it to list
        clipping_post_proc = Clip(high=clip_bounds[1], low=clip_bounds[0])
        self.post_processors.append(clipping_post_proc)

    def _post_process(self, obs):
        """
        A method to sequentially apply post processors to obs.

        Parameters
        ----------
        obs : numpy.array or numpy.ndarray
            An array of values representing the observation space.

        Returns
        -------
        obs : numpy.array or numpy.ndarray
            An array of values representing the observation space.
        """

        for post_processor in self.post_processors:
            obs = post_processor(obs)
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
        # post-process observations
        obs = self._post_process(obs)
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
