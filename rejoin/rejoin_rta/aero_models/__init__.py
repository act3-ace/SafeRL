import abc
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
    def step(self, step_size, state, action=None):
        ...

class DynamicsController(BaseController):
    def __init__(self, dynamics, config=None):
        self.dynamics = dynamics
        self.config = config

    def preprocess_action(self, action):
        return action

    def step(self, step_size, state, action=None):
        control = self.preprocess_action(action)

        next_state = self.dynamics.step(step_size, state, control)

        return next_state

class AgentController(DynamicsController):
 
    def setup_action_space(self, actuators):
        num_actuators = len(actuators)
        action_space_tup = ()
        self.action_vals = []
        for actuator in actuators:
            if actuator.name not in self.config['actuators']:
                raise ValueError("Actuator {} not found in action space config".format(actuator.name))

            actuator_config = self.config['actuators'][actuator.name]
            if actuator_config['space'] == 'discrete':
                vals = np.linspace(actuator.bounds[0], actuator.bounds[1], actuator_config['points'])
                actuator_action_space = Discrete(actuator_config['points'])

            elif actuator_config['space'] == 'continuous':
                vals = None
                actuator_action_space = Box(low=actuator.bounds[0], high=actuator.bounds[1], shape=(1,1))
            else:
                raise ValueError("Action Config for Actuator {} has invalid space of {}. Should be 'continuous' or 'discrete'")

            action_space_tup += (actuator_action_space, )
            self.action_vals.append(vals)

        self.action_space = Tuple(action_space_tup)

    def preprocess_action(self, action):
        return action # TODO

    def step(self, step_size, state, action, dynamics):

        action = self.preprocess_action(action)

        next_state = dynamics.step(state, action)

        return next_state



