"""
This module defines the Platform, State, Dynamics, and Processors needed to simulate a simple 1D Docking environment.

Author: John McCarroll
"""
import gym.spaces
import numpy as np
from scipy.spatial.transform import Rotation

from saferl.aerospace.models.integrators.integrator_1d import BaseIntegrator
from saferl.environment.models.platforms import BasePlatformStateVectorized, ContinuousActuator, BaseActuatorSet,\
                                                BaseLinearODESolverDynamics
from saferl.environment.tasks.processor import ObservationProcessor, StatusProcessor


class Integrator3d(BaseIntegrator):
    """
    This class defines implements the methods and properties necessary for a basic integrator object.

    This class overrides the BaseIntegrator constructor in order to initialize the Dynamics, Actuators, and State
    objects required for a 1D Integrator platform.
    """

    def __init__(self, name, controller=None, integration_method='RK45'):
        dynamics = Integrator3dDynamics(integration_method=integration_method)
        actuator_set = Integrator3dActuatorSet()
        state = Integrator3dState()

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


class Integrator3dState(BasePlatformStateVectorized):
    """
    This class maintains a 6 element vector, which represents the state of the Integrator1D Platform. The two elements
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


class Integrator3dActuatorSet(BaseActuatorSet):
    """
    This class defines the sole actuator required to propel a Integrator1D Platform in a 1D environment.
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


class Integrator3dDynamics(BaseLinearODESolverDynamics):
    """
    This class implements a simplified dynamics model for our 3 dimensional environment.
    """

    def __init__(self, integration_method='RK45'):
        self.m = 1  # kg

        super().__init__(integration_method=integration_method)

    def gen_dynamics_matrices(self):
        A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ], dtype=np.float64)

        B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1 / self.m, 0, 0],
            [0, 1 / self.m, 0],
            [0, 0, 1 / self.m],
        ], dtype=np.float64)

        return A, B


# Processors
class Integrator3dObservationProcessor(ObservationProcessor):
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


class Integrator3dDockingVelocityLimit(StatusProcessor):
    def __init__(self, name, target, dist_status, vel_threshold, threshold_dist, slope=2):
        self.target = target
        self.dist_status = dist_status
        self.vel_threshold = vel_threshold
        self.threshold_dist = threshold_dist
        self.slope = slope
        super().__init__(name)

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        pass

    def _process(self, sim_state):
        dist = sim_state.status[self.dist_status]

        vel_limit = self.vel_threshold

        if dist > self.threshold_dist:
            vel_limit += self.slope * (dist - self.threshold_dist)

        return vel_limit
