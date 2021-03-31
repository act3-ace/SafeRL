import abc

import numpy as np
import math
from scipy.spatial.transform import Rotation

from saferl.environment.models import BasePlatform, BasePlatformStateVectorized, ContinuousActuator, \
    BaseActuatorSet, BaseODESolverDynamics


class BaseDubinsPlatform(BasePlatform):

    @property
    def v(self):
        return self.state.v

    @property
    def yaw(self):
        return self.state.yaw

    @property
    def pitch(self):
        return self.state.pitch

    @property
    def roll(self):
        return self.state.roll

    @property
    def heading(self):
        return self.state.heading

    @property
    def gamma(self):
        return self.state.gamma


class BaseDubinsState(BasePlatformStateVectorized):

    @property
    @abc.abstractmethod
    def v(self):
        ...

    @property
    def velocity(self):
        velocity = np.array([
            self.v * math.cos(self.heading) * math.cos(self.gamma),
            self.v * math.sin(self.heading) * math.cos(self.gamma),
            self.v * math.sin(self.gamma),
        ], dtype=np.float64)
        return velocity

    @property
    def yaw(self):
        return self.heading

    @property
    def pitch(self):
        return self.gamma

    @property
    @abc.abstractmethod
    def roll(self):
        ...

    @property
    @abc.abstractmethod
    def heading(self):
        ...

    @property
    @abc.abstractmethod
    def gamma(self):
        ...


class Dubins2dPlatform(BaseDubinsPlatform):

    def __init__(self, config=None, controller=None, **kwargs):
        # TODO: Set up from config

        dynamics = Dubins2dDynamics()
        actuator_set = Dubins2dActuatorSet()

        state = Dubins2dState()

        super().__init__(dynamics, actuator_set, controller, state, config=config, **kwargs)

    def _generate_info(self):
        info = {
            'state': self.state.vector,
            'x': self.x,
            'y': self.y,
            'heading': self.heading,
            'v': self.v,
        }

        return info


class Dubins2dState(BaseDubinsState):

    def build_vector(self, x=0, y=0, heading=0, v=50, **kwargs):

        return np.array([x, y, heading, v], dtype=np.float64)

    @property
    def x(self):
        return self._vector[0]

    @x.setter
    def x(self, value):
        self._vector[0] = value

    @property
    def y(self):
        return self._vector[1]

    @y.setter
    def y(self, value):
        self._vector[1] = value

    @property
    def z(self):
        return 0

    @property
    def heading(self):
        return self._vector[2]

    @heading.setter
    def heading(self, value):
        self._vector[2] = value

    @property
    def v(self):
        return self._vector[3]

    @v.setter
    def v(self, value):
        self._vector[3] = value

    @property
    def position(self):
        position = np.zeros((3,))
        position[0:2] = self._vector[0:2]
        return position

    @property
    def orientation(self):
        return Rotation.from_euler('z', self.yaw)

    @property
    def gamma(self):
        return 0

    @property
    def roll(self):
        return 0


class Dubins2dActuatorSet(BaseActuatorSet):

    def __init__(self):

        actuators = [
            ContinuousActuator(
                'rudder',
                [np.deg2rad(-6), np.deg2rad(6)],
                0
            ),
            ContinuousActuator(
                'throttle',
                [-10, 10],
                0
            )
        ]

        super().__init__(actuators)


class Dubins2dDynamics(BaseODESolverDynamics):

    def __init__(self, v_min=10, v_max=100, *args, **kwargs):
        self.v_min = v_min
        self.v_max = v_max

        super().__init__(*args, **kwargs)

    def step(self, step_size, state, control):
        state = super().step(step_size, state, control)

        # enforce velocity limits
        if state.v < self.v_min or state.v > self.v_max:
            state.v = max( min(state.v, self.v_max), self.v_min)

        return state

    def dx(self, t, state_vec, control):
        _, _, heading, v = state_vec
        rudder, throttle = control

        # enforce velocity limits
        if v <= self.v_min and throttle < 0:
            throttle = 0
        elif v >= self.v_max and throttle > 0:
            throttle = 0

        x_dot = v * math.cos(heading) # x_dot
        y_dot = v * math.sin(heading) # y_dot
        heading_dot = rudder
        v_dot = throttle

        dx_vec = np.array([x_dot, y_dot, heading_dot, v_dot], dtype=np.float64)

        return dx_vec
