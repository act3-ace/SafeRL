import abc

import gym.spaces
import numpy as np
import math
from collections.abc import Iterable

from saferl.environment.tasks.processor.post_processors import Normalize, Clip, Rotate


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
    def __init__(self,
                 name=None,
                 normalization=None,
                 clip=None,
                 # rotation_reference=None,
                 post_processors=None,
                 observation_space_shape=None):
        """
        The class constructor handles the assignment of member variables and the instantiation of PostProcessors.
        If the 'normalization' kwarg is assigned a list of floats, a default normalization PostProcessor is instantiated
        using the values in the given list as constants.
        If the 'clip' kwarg is assigned a two element list, a default clipping PostProcessor is created, using the first
        element of the given list as the lower bound and the second element as the upper bound.
        If the 'post_processors' kwarg is assigned a list of dict configs, they are instantiated and maintained in
        order.

        NOTE: To avoid redundant applications of PostProcessors, two flags are maintained - has_normalization and
        has_clipping. If a PostProcessor that extends Normalize or Clip is defined in the list of PostProcessor configs,
        the respective flag is mutated. This is to ensure that normalization or clipping defined in the post_processors
        list takes priority over normalization or clipping defined via the 'normalization' or 'clip' kwarg shortcuts.
        While extended this class, if default normalization or clipping are desired, it is recommended that you use the
        proper flags and private factory methods. Here's an example:

        if not self.has_normalization:
            # if no custom normalization defined
            self._add_normalization(LIST_OF_DESIRED_DEFAULT_NORMALIZATION_CONSTANTS)

        Checking the 'has_normalization' flag ensures that no other normalization PostProcessors have been created yet
        and using the '_add_normalization' method handles the creation and insertion of a normalization
        PostProcessor and subsequent setting of  the 'has_normalization' flag. The same applies for the 'has_clipping'
        flag and '_add_clipping' method.


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

        # define post_processor flags
        self.observation_space = None
        self.has_normalization = False
        self.has_clipping = False

        # define Box observation space
        if observation_space_shape:
            # observation space defined in config
            assert type(observation_space_shape) in [int, float]
            self.observation_space = gym.spaces.Box(shape=(observation_space_shape,), low=-math.inf, high=math.inf)
        else:
            # observation space NOT defined in config: delegate to subclass method
            self.observation_space = self.define_observation_space()

        self.normalization = np.array(normalization, dtype=np.float64) if type(normalization) is list else normalization
        self.clip = clip                            # clip[0] == min clip bound, clip[1] == max clip bound
        # self.rotation_reference = rotation_reference
        self.post_processors = []                   # list of PostProcessors

        # create and store post processors
        if isinstance(post_processors, Iterable):
            for post_processor in post_processors:
                assert "class" in post_processor, \
                    "No 'class' key found in {} for construction of PostProcessor.".format(post_processor)
                assert "config" in post_processor, \
                    "No 'config' key found in {} for construction of PostProcessor.".format(post_processor)

                # # check if PostProcessor is form of rotation
                # if self.rotation_reference is not None and issubclass(post_processor_class, Rotate):
                #     raise TypeError(
                #         "Rotation defined in {} config and PostProcessor list. \
                #         Please use PostProcessors to combine multiple rotations".format(self.__name__))

                post_processor_class = post_processor["class"]
                self.post_processors.append(post_processor_class(**post_processor["config"]))

                # check if PostProcessor was normalization or clipping
                if issubclass(post_processor_class, Normalize):
                    self.has_normalization = True
                if issubclass(post_processor_class, Clip):
                    self.has_clipping = True

        # apply postprocessors to Box observation space definition
        for post_processor in self.post_processors:
            self.observation_space = post_processor.modify_observation_space(self.observation_space)
            1+1

        # add norm + clipping postprocessors
        if self.normalization is not None and not self.has_normalization:
            self._add_normalization(self.normalization)
        if self.clip is not None and not self.has_clipping:
            self._add_clipping(self.clip)

    @abc.abstractmethod
    def define_observation_space(self) -> gym.spaces.Box:
        """
        This method shall be used to define the bounds and shape of a 1D observation space to be used by RL agents.

        Returns
        -------
        observation_space : gym.spaces.Box
            This Box sets the shape and upper and lower bounds of each element within a 1D observation space vector.
        """
        raise NotImplementedError

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
        self.has_normalization = True

    def _add_clipping(self, clip_bounds):
        # ensure clip_bounds format
        assert type(clip_bounds) == list, \
            "Expected a list for variable \'clip\', but instead got: {}".format(type(clip_bounds))
        assert 2 == len(clip_bounds), \
            "Expected a list of length 2 for variable \'clip\', but instead got: {}".format(len(clip_bounds))

        # create clipping PostProcessor and add it to list
        clipping_post_proc = Clip(high=clip_bounds[1], low=clip_bounds[0])
        self.post_processors.append(clipping_post_proc)
        self.has_clipping = True

    def _post_process(self, obs, sim_state):
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
            obs = post_processor(obs, sim_state)
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
        obs = self._post_process(obs, sim_state)
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
