"""
This module defines generic ObservationProcessors that may be used to easily construct and customize an agent's
observation space.

Author: John McCarroll
"""

import abc

import numpy as np
import gym
import math
from collections.abc import Iterable

from saferl.environment.tasks.processor import ObservationProcessor
from saferl.environment.tasks.processor.post_processors import Rotate


class SpatialObservationProcessor(ObservationProcessor):
    """
    Class to handle observations in 3D and 2D space.
    """
    def __init__(self, name=None,
                 normalization=None,
                 clip=None,
                 rotation_reference=None,
                 post_processors=None,
                 two_d=False):

        assert type(two_d) == bool, "Expected bool for 'two_d' parameter, but found {}".format(type(two_d))
        self.two_d = two_d

        super().__init__(name=name,
                         normalization=normalization,
                         clip=clip,
                         post_processors=post_processors)

        self.rotation_reference = rotation_reference
        self.has_rotation = False

        # looping over subclasses
        if isinstance(post_processors, Iterable):
            for post_processor in post_processors:
                post_processor_class = post_processor["class"]
                if issubclass(post_processor_class, Rotate):
                    self.has_rotation = True

        if self.rotation_reference is not None and not self.has_rotation:
            self._add_rotation(self.rotation_reference)

    def define_observation_space(self) -> gym.spaces.Box:
        """
        Spatial observation spaces will be 3D by default. An 'two_d' flag enables the return of a 2D positional
        observation space. Observation space bounds are left as negative to positive infinity to allow users maximum
        flexibility.

        Returns
        -------
        observation_space : gym.spaces.Box
            The two to three element vector corresponding to the relative position between two specified environment
            objects.
        """

        shape = (2,) if self.two_d else (3,)
        observation_space = gym.spaces.Box(shape=shape, low=-math.inf, high=math.inf)
        return observation_space

    def _add_rotation(self, rotation_reference):
        # create rotation PostProcessor and add it to list
        rotation_post_proc = Rotate(reference=rotation_reference)
        new_post_processors = [rotation_post_proc]
        # place rotation post proc in front of post proc list
        new_post_processors.extend(self.post_processors)
        self.post_processors = new_post_processors
        self.has_rotation = True

    @abc.abstractmethod
    def _process(self, sim_state) -> np.ndarray:
        raise NotImplementedError


class RelativePositionObservationProcessor(SpatialObservationProcessor):
    """
    Compute and represent the positional difference between two objects.
    """
    def __init__(self, name=None,
                 normalization=None,
                 clip=None,
                 rotation_reference=None,
                 post_processors=None,
                 reference=None,
                 target=None,
                 two_d=False):

        super().__init__(name=name,
                         normalization=normalization,
                         clip=clip,
                         rotation_reference=rotation_reference,
                         post_processors=post_processors,
                         two_d=two_d)
        self.reference = reference
        self.target = target

    def _process(self, sim_state) -> np.ndarray:
        # ensure both objects are within env_objs
        assert self.reference in sim_state.env_objs, \
            "The provided reference object, {}, is not found within the state's environment objects"\
            .format(self.reference)
        assert self.target in sim_state.env_objs, \
            "The provided target object, {}, is not found within the state's environment objects".format(self.target)

        reference = sim_state.env_objs[self.reference]
        target = sim_state.env_objs[self.target]

        # ensure both object have positions
        assert hasattr(reference, "position"), "The provided reference object, {}, has no 'position' attribute!"
        assert hasattr(target, "position"), "The provided target object, {}, has no 'position' attribute!"

        positional_diff = target.position - reference.position

        # apply dimensionality
        if self.two_d:
            positional_diff = positional_diff[0:2]

        return positional_diff


class VelocityObservationProcessor(SpatialObservationProcessor):
    """
    Retrieve and represent the 'velocity' attribute of an environment object
    """
    # collect and store specified attr of specified env_obj
    def __init__(self,
                 name=None,
                 normalization=None,
                 clip=None,
                 rotation_reference=None,
                 post_processors=None,
                 env_object_name=None,
                 two_d=False):

        super().__init__(name=name,
                         normalization=normalization,
                         clip=clip,
                         rotation_reference=rotation_reference,
                         post_processors=post_processors,
                         two_d=two_d)
        self.env_object_name = env_object_name
        self.attribute = "velocity"

    def _process(self, sim_state) -> np.ndarray:
        # ensure both objects are within env_objs
        assert self.env_object_name in sim_state.env_objs, \
            "The provided env_object, {}, is not found within the state's environment objects" \
            .format(self.env_object_name)

        env_object = sim_state.env_objs[self.env_object_name]

        # ensure object has desired attribute
        assert hasattr(env_object, self.attribute), \
            "The provided env_object, {}, has no {} attribute!".format(self.env_object_name, self.attribute)

        value = getattr(env_object, self.attribute)

        # apply dimensionality
        if self.two_d and len(value) > 2:
            value = value[0:2]

        return value
