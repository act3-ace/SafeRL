import abc
import copy
import numpy as np
from gym.spaces import Discrete, Box, Tuple


class BaseActuator(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def space(self) -> str:
        ...

class ContinuousActuator(BaseActuator):
    def __init__(self, name, bounds):
        self._name = name
        self._space = 'continuous'
        self._bounds = bounds

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def space(self) -> str:
        return self._space

    @property
    def bounds(self) -> list:
        return self._bounds

class BaseController(abc.ABC):
    def __init__(self, config = None):
        self.config = config

    @abc.abstractmethod
    def generate_control(self, state, action=None):
        ...

class PassThroughController(BaseController):

    def generate_control(self, state, action=None):
        return action

class AgentController(BaseController):

    def __init__(self, dynamics, config=None):
        self.dynamics = dynamics
        self.config = config

        self.setup_action_space()
 
    def setup_action_space(self):

        actuators = self.dynamics.actuators

        num_actuators = len(actuators)
        action_space_tup = ()
        self.action_preprocessors = []
        self.action_control_mapping = []

        # loop over dynamics actuators and match with actuator configs
        for control_idx, actuator in enumerate(actuators):
            if actuator.name not in self.config['actuators']:
                continue
            else:
                actuator_config = self.config['actuators'][actuator.name]

                if actuator.space == 'continuous':
                    # determine upper and lower bounds of actuator range. Should be the intersection of the actuator object bounds and the actuator config bounds
                    if 'bounds' in actuator_config:
                        bounds_min = max(actuator.bounds[0], actuator_config['bounds'][0])
                        bounds_max = min(actuator.bounds[1], actuator_config['bounds'][1])
                    else:
                        bounds_min = actuator.bounds[0]
                        bounds_max = actuator.bounds[1]

                    if actuator_config['space'] == 'discrete':
                        # if actuator in continuous but config is discrete, discretize actuator bounds
                        vals = np.linspace(bounds_min, bounds_max, actuator_config['points'])
                        preprocessor = ActionPreprocessorDiscreteMap(vals)
                        actuator_action_space = Discrete(actuator_config['points'])

                    elif actuator_config['space'] == 'continuous':
                        # if both actuator and config are continuous, simply pass through value to control
                        preprocessor = ActionPreprocessorPassThrough
                        actuator_action_space = Box(low=bounds_min, high=bounds_max, shape=(1,1))
                    else:
                        raise ValueError("Action Config for Actuator {} has invalid space of {}. Should be 'continuous' or 'discrete'".format(actuator.name, actuator.space))

                elif actuator.space == 'discrete':
                    # if the actuator is discrete, ignore actuator config. Use actuator defined points and pass through value to control
                    preprocessor = ActionPreprocessorPassThrough
                    actuator_action_space = Discrete(actuator.points)
                else:
                    raise ValueError("Actuator {} has invalid space of {}. Should be 'continuous' or 'discrete'".format(actuator.name, actuator.space))

                action_space_tup += (actuator_action_space, )

                # create preprocessor function to process action index input to control input
                self.action_preprocessors.append(preprocessor)

                # map action idx to actuator control idx
                self.action_control_mapping.append(control_idx)

        self.action_space = Tuple(action_space_tup)

    def generate_control(self, state, action=None):
        control = copy.deepcopy(self.dynamics.default_control)

        if action is not None:
            for i, action_val in enumerate(action):
                if action_val is not None:
                    action_val_processed = self.action_preprocessors[i](action_val)
                    control[self.action_control_mapping[i]] = action_val_processed

        return control

class ActionPreprocessor(abc.ABC):
    @abc.abstractmethod
    def __call__(self, action):
        ...

class ActionPreprocessorPassThrough(ActionPreprocessor):
    def __call__(self, action):
        return action

class ActionPreprocessorDiscreteMap(ActionPreprocessor):
    def __init__(self, vals):
        self.vals = vals

    def __call__(self, action):
        return self.vals[action]



