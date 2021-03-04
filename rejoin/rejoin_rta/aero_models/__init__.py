import abc
import copy
import numpy as np
from gym.spaces import Discrete, Box, Tuple
from scipy import integrate

class BaseActuator(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def space(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def default(self):
        ...

class ContinuousActuator(BaseActuator):
    def __init__(self, name, bounds, default):
        self._name = name
        self._space = 'continuous'
        self._bounds = bounds

        if isinstance(default, np.ndarray):
            self._default = default
        else:
            self._default = np.array(default, ndmin=1, dtype=np.float64)

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def space(self) -> str:
        return self._space

    @property
    def bounds(self) -> list:
        return copy.deepcopy(self._bounds)

    @property
    def default(self):
        return copy.deepcopy(self._default)

class BaseController(abc.ABC):
    def __init__(self, config = None):
        self.config = config

    @abc.abstractmethod
    def gen_actuation(self, state, action=None):
        ...

class PassThroughController(BaseController):

    def gen_actuation(self, state, action=None):
        return action

class AgentController(BaseController):

    def __init__(self, actuator_set, config):
        self.config = config
        self.actuator_set = actuator_set
        self.actuator_config_list = self.config['actuators']

        self.setup_action_space()
 
    def setup_action_space(self):

        actuators = self.actuator_set.actuators

        self.action_preprocessors = []
        action_space_tup = ()

        # loop over actuators in controller config and setup preprocessors
        for i, actuator_config in enumerate(self.actuator_config_list):
            actuator_name = actuator_config['name']

            # get associated actuator from platform's actuator_set
            if actuator_name in self.actuator_set.name_idx_map:
                actuator = self.actuator_set.actuators[ self.actuator_set.name_idx_map[actuator_name] ]
            else:
                raise ValueError("Actuator name {} not found in platform's actuator set".format(actuator_name))

            if actuator.space == 'continuous':
                # determine upper and lower bounds of actuator range. Should be the intersection of the actuator object bounds and the actuator config bounds
                if 'bounds' in actuator_config:
                    bounds_min = max(actuator.bounds[0], actuator_config['bounds'][0])
                    bounds_max = min(actuator.bounds[1], actuator_config['bounds'][1])
                else:
                    bounds_min = actuator.bounds[0]
                    bounds_max = actuator.bounds[1]

                if ('space' not in actuator_config) or (actuator_config['space'] == 'continuous'):
                    # if both actuator and config are continuous, simply pass through value to control
                    preprocessor = ActionPreprocessorPassThrough(actuator_name)
                    actuator_action_space = Box(low=bounds_min, high=bounds_max, shape=(1,))
                elif actuator_config['space'] == 'discrete':
                    # if actuator in continuous but config is discrete, discretize actuator bounds
                    vals = np.linspace(bounds_min, bounds_max, actuator_config['points'])
                    preprocessor = ActionPreprocessorDiscreteMap(actuator_name, vals)
                    actuator_action_space = Discrete(actuator_config['points'])
                else:
                    raise ValueError("Action Config for Actuator {} has invalid space of {}. Should be 'continuous' or 'discrete'".format(actuator.name, actuator_config['space']))

            elif actuator.space == 'discrete':
                # if the actuator is discrete, ignore actuator config. Use actuator defined points and pass through value to control
                raise NotImplementedError

            else:
                raise ValueError("Actuator {} has invalid space of {}. Should be 'continuous' or 'discrete'".format(actuator.name, actuator.space))
            
            # append actuator action space and preprocessor
            action_space_tup += (actuator_action_space, )
            self.action_preprocessors.append(preprocessor)


        self.action_space = Tuple(action_space_tup)

    def gen_actuation(self, state, action=None):
        actuation = {}

        if action is not None:
            for i, action_val in enumerate(action):
                if action_val is not None:

                    if not isinstance(action_val, np.ndarray):
                        action_val = np.array(action_val, ndmin=1)

                    actuator_name, action_processed = self.action_preprocessors[i](action_val)
                    actuation[actuator_name] = action_processed
            
        return actuation

class ActionPreprocessor(abc.ABC):
    def __init__(self, name):
        self.name = name
    
    @abc.abstractmethod
    def preprocess(self, action):
        ...

    def __call__(self, action):
        return copy.deepcopy(self.name), self.preprocess(action)

class ActionPreprocessorPassThrough(ActionPreprocessor):
    
    def preprocess(self, action):
        return action

class ActionPreprocessorDiscreteMap(ActionPreprocessor):
   
   
    def __init__(self, name, vals):
        self.vals = vals
        super().__init__(name)

    def preprocess(self, action):
        return self.vals[action]

class BaseActuatorSet:
    def __init__(self, actuators):
        self.actuators = actuators

        self.name_idx_map = {}
        for i, actuator in enumerate(self.actuators):
            self.name_idx_map[actuator.name] = i

    def gen_control(self, actuation=None):
        control_list = []

        if actuation is None:
            actuation = {}

        for actuator in self.actuators:
            actuator_name = actuator.name
            
            if actuator_name in actuation:
                actuator_control = actuation[actuator_name]
            else:
                actuator_control = actuator.default

            control_list.append(actuator_control)

        control = np.concatenate(control_list)

        return control

class BasePlatform(abc.ABC):


    def __init__(self, dynamics, actuator_set, controller, init_dict={}):

        self.dependent_objs = []

        self.dynamics = dynamics
        self.actuator_set = actuator_set
        self.controller = controller

        self.reset(**init_dict)

    def reset(self, **kwargs):
        self.state = self.init_state(**kwargs)
        self.actuation_cur = None
        self.control_cur = None

        for obj in self.dependent_objs:
            obj.reset()

    def step(self, step_size, action=None):
        actuation = self.controller.gen_actuation(self.state, action)

        control = self.actuator_set.gen_control(actuation)

        # TODO save current actuation
        self.actuation_cur = copy.deepcopy(actuation)
        self.control_cur = copy.deepcopy(control)

        self.state = self.dynamics.step(step_size, self.state, control)

        for obj in self.dependent_objs:
            obj.step()

    def register_dependent_obj(self, obj):
        self.dependent_objs.append(obj)

    @abc.abstractmethod
    def init_state(self, **kwargs):
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

        if self.integration_method == "RK45":
            sol = integrate.solve_ivp(self.dx, (0,step_size), state, args=(control,))

            state = sol.y[:,-1] # save last timestep of integration solution 
        elif self.integration_method == 'Euler':
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
        return super().step(step_size, state, control)