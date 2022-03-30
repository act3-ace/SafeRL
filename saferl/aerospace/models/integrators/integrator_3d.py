"""
This module defines the Platform, State, Dynamics, and Processors needed to simulate a simple 1D Docking environment.

Author: John McCarroll
"""
import math

import gym.spaces
import numpy as np
from scipy.spatial.transform import Rotation
import copy

from saferl.aerospace.models.integrators.integrator_1d import BaseSpacecraft
from saferl.environment.models.platforms import BasePlatform, BasePlatformStateVectorized, ContinuousActuator, \
    BaseActuatorSet, BaseLinearODESolverDynamics
from saferl.environment.tasks.processor import ObservationProcessor, StatusProcessor


class Spacecraft3D(BaseSpacecraft):
    """
    This class defines implements the methods and properties necessary for a basic spacecraft object.

    This class overrides the BaseSpacecraft constructor in order to initialize the Dynamics, Actuators, and State
    objects required for a 1D Spacecraft platform.
    """

    def __init__(self, name, controller=None, integration_method='RK45'):
        dynamics = Dynamics3D(integration_method=integration_method)
        actuator_set = ActuatorSet3D()
        state = State3D()

        super().__init__(name, dynamics, actuator_set, state, controller)

    def generate_info(self):
        info = {
            'y_dot': self.y_dot,
            'z_dot': self.z_dot,
        }

        info_parent = super().generate_info()
        info_ret = {**info_parent, **info}

        return info_ret

    @property
    def y_dot(self):
        return self.state.y_dot

    @property
    def y(self):
        return self.state.y

    @property
    def z_dot(self):
        return self.state.z_dot

    @property
    def z(self):
        return self.state.z

    # @property
    # def position(self):
    #     return self.state.position


class State3D(BasePlatformStateVectorized):
    """
    This class maintains a 6 element vector, which represents the state of the Spacecraft1D Platform. The two elements
    of the state vector, in order, are: position on the x axis and velocity along the x axis. This class also defines
    required properties of a BasePlatformStateVectorized child in a form compatible with the rest of the framework.
    """

    def build_vector(self, x=0, y=0, z=0, x_dot=0, y_dot=0, z_dot=0, **kwargs):
        return np.array([x, y, z, x_dot, y_dot, z_dot], dtype=np.float64)

    @property
    def x(self):
        return self._vector[0]

    @property
    def y(self):
        return self._vector[1]

    @property
    def z(self):
        return self._vector[2]

    @property
    def x_dot(self):
        return self._vector[3]

    @property
    def y_dot(self):
        return self._vector[4]

    @property
    def z_dot(self):
        return self._vector[5]

    @property
    def position(self):
        position = self._vector[0:3]
        return position

    @property
    def orientation(self):
        # always return a no rotation quaternion as points do not have an orientation
        return Rotation.identity()

    @property
    def velocity(self):
        vel = np.array([self.x_dot, self.y_dot, self.z_dot], dtype=np.float64)
        return vel

    @property
    def velocity_mag(self):
        vel = np.array([self.x_dot, self.y_dot, self.z_dot], dtype=np.float64)
        vel_mag = np.linalg.norm(vel)
        return vel_mag


class ActuatorSet3D(BaseActuatorSet):
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
            ContinuousActuator(
                'thrust_y',
                [-1, 1],
                0
            ),
            ContinuousActuator(
                'thrust_z',
                [-1, 1],
                0
            ),
        ]

        super().__init__(actuators)


class Dynamics3D(BaseLinearODESolverDynamics):
    """
    This class implements a simplified dynamics model for our 1 dimensional environment.
    """

    def __init__(self, integration_method='RK45'):
        self.m = 1  # kg

        super().__init__(integration_method=integration_method)

    def gen_dynamics_matrices(self):
        m = self.m
        n = 0.1

        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3 * n ** 2, 0, 0, 0, 2 * n, 0],
            [0, 0, 0, -2 * n, 0, 0],
            [0, 0, -n ** 2, 0, 0, 0],
        ], dtype=np.float64)

        B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1 / m, 0, 0],
            [0, 1 / m, 0],
            [0, 0, 1 / m],
        ], dtype=np.float64)

        return A, B


# Processors
class Docking3dObservationProcessor(ObservationProcessor):
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

        observation_space = gym.spaces.Box(low=low, high=high, shape=(6,))

        return observation_space

    def _process(self, sim_state):
        obs = np.copy(sim_state.env_objs[self.deputy].state.vector)
        return obs


class Docking3dVelocityLimit(StatusProcessor):
    def __init__(self, name, dist_status):
        self.dist_status = dist_status
        super().__init__(name)

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        pass

    def _process(self, sim_state):
        dist = sim_state.status[self.dist_status]  # TODO: get dist from deputy state?

        vel_limit = math.sqrt(2 * dist)

        return vel_limit


class Docking3dFailureStatusProcessor(StatusProcessor):
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
        y = sim_state.env_objs[self.deputy].y
        z = sim_state.env_objs[self.deputy].z

        if self.time_elapsed > self.timeout:
            failure = 'timeout'
        elif sim_state.status[self.docking_distance] >= self.max_goal_distance:
            failure = 'distance'
        elif not sim_state.status[self.max_vel_constraint_status] or x > 0 or y > 0 or z > 0:
            failure = 'crash'
        else:
            failure = False
        return failure


class Docking3dVelocityLimitCompliance(StatusProcessor):
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

        if isinstance(target_obj, Spacecraft3D) and isinstance(ref_obj, Spacecraft3D):
            target_vel = target_obj.state.velocity_mag
            ref_vel = ref_obj.state.velocity_mag

            vel_limit = sim_state.status[self.vel_limit_status]

            rel_vel = target_vel - ref_vel

            compliance = vel_limit - rel_vel
            return compliance
        else:
            raise(ValueError, "chief and deputy environment objects must be of type Spacecraft3D")


class Docking3dRelativeVelocityConstraint(StatusProcessor):
    def __init__(self, name, vel_limit_compliance_status):
        self.vel_limit_compliance_status = vel_limit_compliance_status
        super().__init__(name)

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        pass

    def _process(self, sim_state):
        return 0 <= sim_state.status[self.vel_limit_compliance_status]