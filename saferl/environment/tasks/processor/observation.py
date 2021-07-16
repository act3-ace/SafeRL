"""
This module defines generic ObservationProcessors that may be used to easily construct and customize an agent's
observation space.

Author: John McCarroll
"""

import numpy as np
from saferl.environment.tasks.processor import ObservationProcessor


class RelativePositionObservationProcessor(ObservationProcessor):
    """"
    TODO:
    Find and transform positions of two objects.
    """
    def __init__(self, name=None,
                 normalization=None,
                 clip=None,
                 post_processors=None,
                 reference=None,
                 target=None):
        super().__init__(name=name, normalization=normalization, clip=clip, post_processors=post_processors)
        self.reference = reference
        self.target = target

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

        absolute_position = target.position - reference.position
        # TODO: was rotation meant to be a post processor?
        relative_rotation = reference.orientation.inv()
        relative_position = relative_rotation.apply(absolute_position)

        relative_position = np.array(relative_position)

        return relative_position


class EnvironmentObjectAttributeObservationProcessor(ObservationProcessor):
    # collect and store specified attr of specified env_obj
    def __init__(self,
                 name=None,
                 normalization=None,
                 clip=None,
                 post_processors=None,
                 env_object=None,
                 attribute_name=None):
        super().__init__(name=name, normalization=normalization, clip=clip, post_processors=post_processors)
        assert type(attribute_name) == str
        self.env_object = env_object
        self.attribute = attribute_name

    def _process(self, sim_state) -> np.ndarray:
        # ensure both objects are within env_objs
        assert self.env_object in sim_state.env_objs, \
            "The provided env_object, {}, is not found within the state's environment objects" \
            .format(self.env_object)
        assert self.attribute in sim_state.env_objs, \
            "The provided attribute, {}, is not found within the state's environment objects" \
            .format(self.attribute)

        env_object = sim_state.env_objs[self.env_object]

        # ensure object has desired attribute
        assert self.attribute in env_object.__dict__, \
            "The provided env_object, {}, has no {} attribute!".format(self.env_object, self.attribute)

        value = env_object.__dict__[self.attribute]

        value = np.array([value])

        return value


"""
NOTES:

- postprocessor parent and child classes should be moved to postprocess.py module in processor package?
- want to separate Rotation into its own postprocessor
- unsure if ObjectAttribute Processor is useful...
"""
