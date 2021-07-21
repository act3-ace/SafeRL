"""
This module defines generic ObservationProcessors that may be used to easily construct and customize an agent's
observation space.

Author: John McCarroll
"""

import numpy as np
import gym
import math
from saferl.environment.tasks.processor import ObservationProcessor


class RelativePositionObservationProcessor(ObservationProcessor):
    """
    Compute and represent the positional difference between two objects.
    """
    def __init__(self, name=None,
                 normalization=None,
                 clip=None,
                 post_processors=None,
                 reference=None,
                 target=None,
                 is_2d=False):
        super().__init__(name=name, normalization=normalization, clip=clip, post_processors=post_processors)
        self.reference = reference
        self.target = target
        assert type(is_2d) == bool, "Expected bool for 'is_2d' parameter, but found {}".format(type(is_2d))
        self.is_2d = is_2d

    def _process(self, sim_state) -> np.ndarray:
        # ensure both objects are within env_objs
        assert self.reference in sim_state.env_objs, \
            "The provided reference object, {}, is not found within the state's environment objects"\
            .format(self.reference)
        assert self.target in sim_state.env_objs, \
            "The provided target object, {}, is not found within the state's environment objects"\
            .format(self.target)

        reference = sim_state.env_objs[self.reference]
        target = sim_state.env_objs[self.target]

        # ensure both object have positions
        assert "position" in reference.__dict__, "The provided reference object, {}, has no 'position' attribute!"
        assert "position" in target.__dict__, "The provided target object, {}, has no 'position' attribute!"

        positional_diff = target.position - reference.position

        # # encapsulate rotation as a post processor
        # relative_rotation = reference.orientation.inv()
        # relative_position = relative_rotation.apply(absolute_position)
        # relative_position = np.array(relative_position)
        # return relative_position

        return positional_diff

    def define_observation_space(self) -> gym.spaces.Box:
        """
        Positional observation spaces will be 3D by default. An 'is_2d' flag enables the return of a 2D positional
        observation space. Observation space bounds are left as negative to positive infinity to allow users maximum
        flexibility.

        Returns
        -------
        observation_space : gym.spaces.Box
            The two to three element vector corresponding to the relative position between two specified environment
            objects.
        """

        shape = (2,) if self.is_2d else (3,)
        observation_space = gym.spaces.Box(shape=shape, low=-math.inf, high=math.inf)
        return observation_space


class VelocityObservationProcessor(ObservationProcessor):
    """
    Retrieve and represent the 'velocity' attribute of an environment object
    """
    # collect and store specified attr of specified env_obj
    def __init__(self,
                 name=None,
                 normalization=None,
                 clip=None,
                 post_processors=None,
                 env_object_name=None,
                 is_2d=False):
        super().__init__(name=name, normalization=normalization, clip=clip, post_processors=post_processors)
        self.env_object_name = env_object_name
        self.attribute = "velocity"
        assert type(is_2d) == bool, "Expected bool for 'is_2d' parameter, but found {}".format(type(is_2d))
        self.is_2d = is_2d

    def _process(self, sim_state) -> np.ndarray:
        # ensure both objects are within env_objs
        assert self.env_object_name in sim_state.env_objs, \
            "The provided env_object, {}, is not found within the state's environment objects" \
            .format(self.env_object_name)
        assert self.attribute in sim_state.env_objs, \
            "The provided attribute, {}, is not found within the state's environment objects" \
            .format(self.attribute)

        env_object = sim_state.env_objs[self.env_object_name]

        # ensure object has desired attribute
        assert self.attribute in env_object.__dict__, \
            "The provided env_object, {}, has no {} attribute!".format(self.env_object_name, self.attribute)

        value = env_object.__dict__[self.attribute]

        value = np.array([value])

        return value

    def define_observation_space(self) -> gym.spaces.Box:
        """
        TODO

        Returns
        -------
        observation_space : gym.spaces.Box
            The two to three element vector corresponding to the velocity of the specified environment object.
        """

        shape = (2,) if self.is_2d else (3,)
        observation_space = gym.spaces.Box(shape=shape, low=-math.inf, high=math.inf)
        return observation_space
