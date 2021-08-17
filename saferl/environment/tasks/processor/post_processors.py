"""
This module defines the abstract PostProcessor class and several generic, concrete subclasses.

Author: John McCarroll
"""

import math
import abc
import numpy as np
import gym


class PostProcessor:
    @abc.abstractmethod
    def __call__(self, input_array, sim_state):
        """
        Subclasses should implement this method to apply post-processing to processor's return values.

        Parameters
        ----------
        input_array : numpy.ndarray
            The value passed to a given post processor for modification.

        Returns
        -------
        input_array
            The modified (processed) input value
        """
        raise NotImplementedError

    @abc.abstractmethod
    def modify_observation_space(self, obs_space: gym.spaces.Box):
        """
        Subclasses shall implement this method to adjust the dimensions of the provided observation space as needed for
        their respective post processing operation.

        Parameters
        ----------
        obs_space : gym.spaces.Box
            The Box representing the observation space available to the RL agent.

        Returns
        -------
        obs_space : gym.spaces.Box
            The modified Box representing the observation space available to the RL agent.
        """
        raise NotImplementedError


class Normalize(PostProcessor):
    def __init__(self, mu=0, sigma=1):
        # ensure mu and sigma compatible types
        acceptable_types = [float, int, list, np.ndarray]
        assert type(mu) in acceptable_types, \
            "Expected kwarg \'mu\' to be type int, float, list, or numpy.ndarray, but received {}" \
            .format(type(mu))
        assert type(sigma) in acceptable_types, \
            "Expected kwarg \'sigma\' to be type int, float, list, or numpy.ndarray, but received {}" \
            .format(type(sigma))

        # convert lists to numpy arrays
        if type(mu) == list:
            mu = np.array(mu)
        if type(sigma) == list:
            sigma = np.array(sigma)

        # store values
        self.mu = mu
        self.sigma = sigma

    def __call__(self, input_array, sim_state):
        # ensure input_array is numpy array
        assert type(input_array) == np.ndarray, \
            "Expected \'input_array\' to be type numpy.ndarray, but instead received {}.".format(type(input_array))

        # check that dims line up for mu and sigma (or that they're scalars)
        if type(self.mu) == np.ndarray:
            assert input_array.shape == self.mu.shape, \
                "Incompatible shapes for \'input_array\' and \'mu\': {} vs {}".format(input_array, self.mu)
        if type(self.sigma) == np.ndarray:
            assert input_array.shape == self.sigma.shape, \
                "Incompatible shapes for \'input_array\' and \'sigma\': {} vs {}".format(input_array, self.sigma)

        # apply normalization
        input_array = np.subtract(input_array, self.mu)
        input_array = np.divide(input_array, self.sigma)

        return input_array

    def modify_observation_space(self, obs_space: gym.spaces.Box):
        # obs_space dimensions not altered by normalization
        # TODO: alter high and low to (-1:1) b/c normalized?
        return obs_space


class Clip(PostProcessor):
    def __init__(self, low=-1, high=1):
        # ensure bounds correct types
        assert type(low) in [int, float], \
            "Expected kwarg \'low\' to be type int or float, but instead received {}".format(type(low))
        assert type(high) in [int, float], \
            "Expected kwarg \'high\' to be type int or float, but instead received {}".format(type(high))
        # ensure correct relation
        assert low < high, "Expected value of variable \'low\' to be less than variable \'high\'"

        # store values
        self.low = low
        self.high = high

    def __call__(self, input_array, sim_state):
        # ensure input_array is numpy array
        assert type(input_array) == np.ndarray, \
            "Expected \'input_array\' to be type numpy.ndarray, but instead received {}.".format(type(input_array))

        # apply clipping in specified range
        input_array = np.clip(input_array, self.high, self.low)

        return input_array

    def modify_observation_space(self, obs_space: gym.spaces.Box):
        # # adjust bounds of obs_space values
        # size = obs_space.shape[0]          # assumes 1d obs space
        # low_array = [self.low] * size
        # obs_space.low = np.array(low_array)
        #
        # size = obs_space.shape[0]          # assumes 1d obs space
        # high_array = [self.high] * size
        # obs_space.low = np.array(high_array)

        return obs_space


class Rotate(PostProcessor):
    def __init__(self, reference=None):
        self.reference = reference

    def __call__(self, input_array, sim_state):
        # assumes the entire input_array is positional info*

        # ensure reference in environment
        assert self.reference in sim_state.env_objs, \
            "The provided reference object, {}, is not found within the state's environment objects"\
            .format(self.reference)

        # check input dims
        input_is_2d = False
        if len(input_array) == 2:
            input_is_2d = True
            input_array = np.concatenate([input_array, [0]])
        assert len(input_array) == 3, \
            "Three dimensional input expected for rotation, but received: {}".format(input_array)

        # apply rotation
        reference = sim_state.env_objs[self.reference]
        relative_rotation = reference.orientation.inv()
        input_array = relative_rotation.apply(input_array)

        # restore correct dimensions
        input_array = input_array[0:2] if input_is_2d else input_array

        return input_array

    def modify_observation_space(self, obs_space: gym.spaces.Box):
        # obs_space dimensions not altered by rotation
        return obs_space


class MagNorm(PostProcessor):
    def __call__(self, input_array, sim_state):
        # assumes entire input_array is the standard repr (aka "rect")

        norm = np.linalg.norm(input_array)
        mag_norm_array = np.concatenate(([norm], input_array / norm))

        return mag_norm_array

    def modify_observation_space(self, obs_space: gym.spaces.Box):
        # find bounds of magnorm of obs space
        min_magnorm = 0
        max_magnorm = np.linalg.norm(obs_space.high)

        # set obs space
        size = obs_space.shape[0]           # assumes 1d obs space
        low = [-1] * size
        high = [1] * size
        obs_space.low = np.concatenate([min_magnorm, low])
        obs_space.high = np.concatenate([max_magnorm, high])

        return obs_space


class DefineBounds(PostProcessor):
    def __init__(self, high=math.inf, low=-math.inf):
        # convert lists to numpy.ndarray
        if type(high) is list:
            high = np.array(high)
        if type(low) is list:
            low = np.array(low)

        acceptable_types = [int, float, np.ndarray]
        assert type(high) in acceptable_types, \
            "Expected parameter 'high' to be of type int, float, list, or numpy.ndarray, but instead received {}"\
            .format(type(high))
        assert type(low) in acceptable_types, \
            "Expected parameter 'low' to be of type int, float, list, or numpy.ndarray, but instead received {}"\
            .format(type(low))
        # TODO: defensive checks to ensure lows are actually less than highs
        #       defensive check to ensure high and low are same dim (if both arrays)

        self.high = high
        self.low = low
        self.high_is_scalar = True if type(self.high) in [int, float] else False
        self.low_is_scalar = True if type(self.low) in [int, float] else False

    def __call__(self, input_array, sim_state):
        return input_array

    def modify_observation_space(self, obs_space: gym.spaces.Box):
        # applies given bounds to received observation space
        # assumes obs_space is 1D array

        # convert scalars to numpy.ndarray
        if self.high_is_scalar:
            high_array = [self.high] * obs_space.shape[0]
            self.high = np.array(high_array)
        if self.low_is_scalar:
            low_array = [self.low] * obs_space.shape[0]
            self.low = np.array(low_array)

        # get size of bounds
        size = self.low.shape[0]

        if obs_space.shape[0] == size:
            obs_space.high = self.high
            obs_space.low = self.low
        else:
            raise ValueError("The shape of the given bounds do not align with the shape of the observation space!")
