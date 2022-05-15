"""
This module defines the Platform, State, Dynamics, and Processors needed to simulate a simple 1D Docking environment.

Author: John McCarroll
"""
import math

import gym.spaces
import numpy as np
from scipy.spatial.transform import Rotation

from saferl.environment.models.platforms import BasePlatform, BasePlatformStateVectorized, ContinuousActuator, \
    BaseActuatorSet, BaseLinearODESolverDynamics
from saferl.environment.tasks.processor import ObservationProcessor, StatusProcessor


class BaseIntegrator(BasePlatform):
    """
    This class defines implements the methods and properties necessary for a basic spacecraft object.
    """

    def generate_info(self):
        info = {
            'state': self.state.vector,
            'x_dot': self.x_dot,
        }

        info_parent = super().generate_info()
        info_ret = {**info_parent, **info}

        return info_ret

    @property
    def x_dot(self):
        return self.state.x_dot

    @property
    def x(self):
        return self.state.x

    @property
    def m(self):
        return self.dynamics.m

class Integrator1d(BaseIntegrator):
    """
    This class overrides the BaseSpacecraft constructor in order to initialize the Dynamics, Actuators, and State
    objects required for a 1D Spacecraft platform.
    """

    def __init__(self, name, controller=None, integration_method='Euler'):

        dynamics = Integrator1dDynamics(integration_method=integration_method)
        actuator_set = Integrator1dActuatorSet()
        state = Integrator1dState()

        super().__init__(name, dynamics, actuator_set, state, controller)



class Integrator1dState(BasePlatformStateVectorized):
    """
    This class maintains a 2 element vector, which represents the state of the Spacecraft1D Platform. The two elements
    of the state vector, in order, are: position on the x axis and velocity along the x axis. This class also defines
    required properties of a BasePlatformStateVectorized child in a form compatible with the rest of the framework.
    """

    def build_vector(self, x=0, x_dot=0, **kwargs):
        return np.array([x, x_dot], dtype=np.float64)

    @property
    def x(self):
        return self._vector[0]

    @property
    def x_dot(self):
        return self._vector[1]

    @property
    def position(self):
        position = np.zeros((3,))
        position[0] = self._vector[0]
        return position

    @property
    def orientation(self):
        # always return a no rotation quaternion as 1D objects do not have an orientation
        return Rotation.identity()

    @property
    def velocity(self):
        vel = np.array([self.x_dot, 0, 0], dtype=np.float64)
        return vel

    @property
    def y(self):
        return 0

    @property
    def z(self):
        return 0


class Integrator1dActuatorSet(BaseActuatorSet):
    """
    This class defines the sole actuator required to propel a Spacecraft1D Platform in a 1D environment.
    """

    def __init__(self):
        actuators = [
            ContinuousActuator(
                'thrust_x',
                [-1, 1],
                0
            ),
        ]

        super().__init__(actuators)



class Integrator1dDynamics(BaseLinearODESolverDynamics):
    """
    This class implements a simplified dynamics model for our 1 dimensional environment.
    """

    def __init__(self, integration_method='RK45'):
        self.m = 1  # kg

        super().__init__(integration_method=integration_method)

    def gen_dynamics_matrices(self):

        A = np.array([
            [0, 1],
            [0, 0],
        ], dtype=np.float64)

        B = np.array([
            [0],
            [1 / self.m],
        ], dtype=np.float64)

        return A, B


# Processors
class Integrator1dObservationProcessor(ObservationProcessor):
    """
    This class defines our 1 dimensional agent's observation space as a simple two element array (containing one
    position value and one velocity value). These two values are retrieved from the state vector of our deputy's state
    vector.
    """

    def __init__(self, name=None, deputy=None, normalization=None, clip=None, post_processors=None):
        # Initialize member variables from config

        # not platform ref, string for name of deputy
        self.deputy = deputy

        # TODO: add default normalization?

        # Invoke parent's constructor
        super().__init__(name=name, normalization=normalization, clip=clip, post_processors=post_processors)

    def define_observation_space(self) -> gym.spaces.Box:
        low = np.finfo(np.float32).min
        high = np.finfo(np.float32).max

        observation_space = gym.spaces.Box(low=low, high=high, shape=(2,))

        return observation_space

    def _process(self, sim_state):
        obs = np.copy(sim_state.env_objs[self.deputy].state.vector)
        return obs


class Integrator1dDockingVelocityLimit(StatusProcessor):
    def __init__(self, name, dist_status):
        self.dist_status = dist_status
        super().__init__(name)

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        pass

    def _process(self, sim_state):
        dist = sim_state.status[self.dist_status]           # TODO: get dist from deputy state?

        vel_limit = math.sqrt(2 * dist)

        return vel_limit


class Docking1dFailureStatusProcessor(StatusProcessor):
    def __init__(self,
                 name,
                 deputy,
                 docking_distance,
                 max_goal_distance,
                 max_vel_constraint_status,
                 timeout):
        super().__init__(name=name)
        self.timeout = timeout
        self.docking_distance = docking_distance
        self.max_goal_distance = max_goal_distance
        self.deputy = deputy
        # self.in_docking_status = in_docking_status
        self.max_vel_constraint_status = max_vel_constraint_status

    def reset(self, sim_state):
        self.time_elapsed = 0

    def _increment(self, sim_state, step_size):
        # increment internal state
        self.time_elapsed += step_size

    def _process(self, sim_state):
        # process state and return status
        x = sim_state.env_objs[self.deputy].x
        
        if self.time_elapsed > self.timeout:
            failure = 'timeout'
        elif sim_state.status[self.docking_distance] >= self.max_goal_distance:
            failure = 'distance'
        elif not sim_state.status[self.max_vel_constraint_status] or x > 0:
            failure = 'crash'
        else:
            failure = False
        return failure


class Integrator1dDockingVelocityLimitCompliance(StatusProcessor):
    def __init__(self, name, target, ref, vel_limit_status):
        self.target = target
        self.ref = ref
        self.vel_limit_status = vel_limit_status
        super().__init__(name)

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        pass

    def _process(self, sim_state):
        target_obj = sim_state.env_objs[self.target]
        ref_obj = sim_state.env_objs[self.ref]

        target_vel = target_obj.x_dot
        ref_vel = ref_obj.x_dot

        vel_limit = sim_state.status[self.vel_limit_status]

        rel_vel = target_vel - ref_vel

        compliance = vel_limit - rel_vel
        return compliance


class Docking1dRelativeVelocityConstraint(StatusProcessor):
    def __init__(self, name, vel_limit_compliance_status):
        self.vel_limit_compliance_status = vel_limit_compliance_status
        super().__init__(name)

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        pass

    def _process(self, sim_state):
        return 0 <=  sim_state.status[self.vel_limit_compliance_status]