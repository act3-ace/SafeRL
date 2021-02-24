import abc
import copy
import numpy as np
from gym.spaces import Discrete, Box, Tuple
from scipy import integrate

class BaseActuatorManager(abc.ABC):


    @property
    @abc.abstractmethod
    def default_control(self):
        ...

    @property
    @abc.abstractmethod
    def actuators(self):
        ...

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
                        preprocessor = ActionPreprocessorPassThrough()
                        actuator_action_space = Box(low=bounds_min, high=bounds_max, shape=(1,))
                    else:
                        raise ValueError("Action Config for Actuator {} has invalid space of {}. Should be 'continuous' or 'discrete'".format(actuator.name, actuator.space))

                elif actuator.space == 'discrete':
                    # if the actuator is discrete, ignore actuator config. Use actuator defined points and pass through value to control
                    raise NotImplementedError
                    preprocessor = None
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


class BasePlatform(abc.ABC):


    def __init__(self, dynamics, actuator_manager, controller, init_dict=None):

        self.dependent_objs = []

        self.dynamics = dynamics
        self.actuator_manager = actuator_manager
        self.controller = controller

        self.reset(init_dict)

    def reset(self, init_dict):
        self.state = self.init_state(init_dict)
        self.actuation_cur = None
        self.control_cur = None

    def step(self, step_size, action=None):
        actuation = self.controller.gen_actuation(self.state, action)

        control = self.actuator_manager.gen_control(actuation)

        # TODO save current actuation
        self.control_cur = np.copy(control)

        self.state = self.compute_step(self.state, control)

    @abc.abstractmethod
    def compute_step(self, state, control):
        ...

    @abc.abstractmethod
    def init_state(self, init_dict):
        ...

class BaseDynamics(abc.ABC):
    
    @abc.abstractmethod
    def step(self, step_size, state, control):
        ...

class BaseODESolverDynmaics(BaseDynamics):


    def __init__(self, integration_method='RK45'):
        self.integration_method = integration_method
        super().__init__()

    @abc.abstractmethod
    def dx(self, t, state, control):
        ...

    def step(self, step_size, state, control):

        if self.integration_method == "rk45":
            sol = integrate.solve_ivp(self.dx, (0,step_size), state, args=(control,))

            state = sol.y[:,-1] # save last timestep of integration solution 
        elif self.integration_method == 'euler': # euler
            state_dot = self.dx(0, state, control)
            state = state + step_size * state_dot
        else:
            raise ValueError("invalid integration method '{}'".format(self.integration_method))

        return state

class BaseLinearODESolverDynamics(BaseODESolverDynmaics):


    def __init__(self, integration_method='RK45'):
        self.A, self.B = self.gen_dynamics_matrices()
        super().__init__(integration_method=integration_method)

    @abc.abstractmethod
    def gen_dynamics_matrices(self):
        ...

    def update_dynamics_matrices(self):
        pass

    def dx(self, t, state, control):
        dx = np.matmul(self.A, state) + np.matmul(self.B, control)
        return dx

    def step(self, step_size, state, control):
        self.update_dynamics_matrices()
        super().step(self, step_size, state, control)