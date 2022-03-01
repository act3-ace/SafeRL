import gym.spaces
import numpy as np
from scipy.spatial.transform import Rotation
import copy

from saferl.environment.models.platforms import BasePlatform, BasePlatformStateVectorized, ContinuousActuator, \
    BaseActuatorSet, BaseLinearODESolverDynamics
from saferl.environment.tasks.processor import ObservationProcessor, StatusProcessor


class BaseSpacecraft(BasePlatform):

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


class Spacecraft1D(BaseSpacecraft):

    def __init__(self, name, controller=None, integration_method='Euler'):

        dynamics = Dynamics1D(integration_method=integration_method)
        actuator_set = ActuatorSet1D()
        state = State1D()

        super().__init__(name, dynamics, actuator_set, state, controller)



class State1D(BasePlatformStateVectorized):

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
        # always return a no rotation quaternion as points do not have an orientation
        return Rotation.identity()

    @property
    def velocity(self):
        vel = np.array([self.x_dot, 0, 0], dtype=np.float64)    # array sizes fixed??*
        return vel

    @property
    def y(self):
        return 0

    @property
    def z(self):
        return 0


class ActuatorSet1D(BaseActuatorSet):

    def __init__(self):
        actuators = [
            ContinuousActuator(
                'thrust_x',
                [-1, 1],
                0
            ),
        ]

        super().__init__(actuators)



class Dynamics1D(BaseLinearODESolverDynamics):
    def __init__(self, m=12, integration_method='Euler'):
        self.m = m  # kg

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
class Docking1dObservationProcessor(ObservationProcessor):
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


class Docking1dFailureStatusProcessor(StatusProcessor):
    def __init__(self,
                 name,
                 docking_distance,
                 max_goal_distance,
                 in_docking_status,
                 timeout):
        super().__init__(name=name)
        self.timeout = timeout
        self.docking_distance = docking_distance
        self.max_goal_distance = max_goal_distance
        self.in_docking_status = in_docking_status

    def reset(self, sim_state):
        self.time_elapsed = 0

    def _increment(self, sim_state, step_size):
        # increment internal state
        self.time_elapsed += step_size

    def _process(self, sim_state):
        # process state and return status
        if self.time_elapsed > self.timeout:
            failure = 'timeout'
        elif sim_state.status[self.docking_distance] >= self.max_goal_distance:
            failure = 'distance'
        # elif sim_state.status[self.in_docking_status] and (not sim_state.status[self.max_vel_constraint_status]):
        #     failure = 'crash'
        else:
            failure = False
        return failure


class Docking1dSuccessStatusProcessor(StatusProcessor):
    def __init__(self, name, in_docking_status):
        super().__init__(name=name)
        self.in_docking_status = in_docking_status

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        # status derived directly from simulation state, therefore no state machine needed
        pass

    def _process(self, sim_state):
        # process stare and return status
        success = sim_state.status[self.in_docking_status]
        return success