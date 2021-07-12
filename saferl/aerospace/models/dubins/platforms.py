import abc

import numpy as np
import math
from scipy.spatial.transform import Rotation

from saferl.environment.models.platforms import BasePlatform, BasePlatformStateVectorized, ContinuousActuator, \
    BaseActuatorSet, BaseODESolverDynamics


class BaseDubinsPlatform(BasePlatform):

    def generate_info(self):
        info = {
            'state': self.state.vector,
            'heading': self.heading,
            'v': self.v,
        }

        info_parent = super().generate_info()
        info_ret = {**info_parent, **info}

        return info_ret

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
        raise NotImplementedError

    @property
    def velocity(self):
        velocity = np.array([
            self.v * math.cos(self.heading) * math.cos(self.gamma),
            self.v * math.sin(self.heading) * math.cos(self.gamma),
            -1 * self.v * math.sin(self.gamma),
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
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def heading(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def gamma(self):
        raise NotImplementedError


class Dubins2dPlatform(BaseDubinsPlatform):

    def __init__(self, controller=None, v_min=10, v_max=100):

        dynamics = Dubins2dDynamics(v_min=v_min, v_max=v_max)
        actuator_set = Dubins2dActuatorSet()

        state = Dubins2dState()

        super().__init__(dynamics, actuator_set, state, controller)


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
            state.v = max(min(state.v, self.v_max), self.v_min)

        return state

    def dx(self, t, state_vec, control):
        _, _, heading, v = state_vec
        rudder, throttle = control

        # enforce velocity limits
        if v <= self.v_min and throttle < 0:
            throttle = 0
        elif v >= self.v_max and throttle > 0:
            throttle = 0

        x_dot = v * math.cos(heading)  # x_dot
        y_dot = v * math.sin(heading)  # y_dot
        heading_dot = rudder
        v_dot = throttle

        dx_vec = np.array([x_dot, y_dot, heading_dot, v_dot], dtype=np.float64)

        return dx_vec


"""
3D Dubins Implementation
"""


class Dubins3dPlatform(BaseDubinsPlatform):

    def __init__(self, controller=None, v_min=10, v_max=100):

        dynamics = Dubins3dDynamics(v_min=v_min, v_max=v_max)
        actuator_set = Dubins3dActuatorSet()
        state = Dubins3dState()

        super().__init__(dynamics, actuator_set, state, controller)

    def generate_info(self):
        info = {
            'gamma': self.gamma,
            'roll': self.roll,
        }

        info_parent = super().generate_info()
        info_ret = {**info_parent, **info}

        return info_ret


class Dubins3dState(BaseDubinsState):

    def build_vector(self, x=0, y=0, z=0, heading=0, gamma=0, roll=0, v=100, **kwargs):
        return np.array([x, y, z, heading, gamma, roll, v], dtype=np.float64)

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
        return self._vector[2]

    @z.setter
    def z(self, value):
        self._vector[2] = value

    @property
    def heading(self):
        return self._vector[3]

    @heading.setter
    def heading(self, value):
        self._vector[3] = value

    @property
    def gamma(self):
        return self._vector[4]

    @gamma.setter
    def gamma(self, value):
        self._vector[4] = value

    @property
    def roll(self):
        return self._vector[5]

    @roll.setter
    def roll(self, value):
        self._vector[5] = value

    @property
    def v(self):
        return self._vector[6]

    @v.setter
    def v(self, value):
        self._vector[6] = value

    @property
    def position(self):
        position = np.zeros((3,))
        position[0:3] = self._vector[0:3]
        return position

    @property
    def orientation(self):
        return Rotation.from_euler('ZYX', [self.yaw, self.pitch, self.roll])


class Dubins3dActuatorSet(BaseActuatorSet):

    def __init__(self):

        actuators = [
            ContinuousActuator(
                'ailerons',
                [np.deg2rad(-6), np.deg2rad(6)],
                0
            ),
            ContinuousActuator(
                'elevator',
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


class Dubins3dDynamics(BaseODESolverDynamics):

    def __init__(self, v_min=10, v_max=100, roll_min=-math.pi/2, roll_max=math.pi/2, g=32.17, *args, **kwargs):
        self.v_min = v_min
        self.v_max = v_max
        self.roll_min = roll_min
        self.roll_max = roll_max
        self.g = g

        super().__init__(*args, **kwargs)

    def step(self, step_size, state, control):
        state = super().step(step_size, state, control)

        # enforce velocity limits
        if state.v < self.v_min or state.v > self.v_max:
            state.v = max(min(state.v, self.v_max), self.v_min)

        # enforce roll limits
        if state.roll < self.roll_min or state.roll > self.roll_max:
            state.roll = max(min(state.roll, self.roll_max), self.roll_min)

        return state

    def dx(self, t, state_vec, control):
        x, y, z, heading, gamma, roll, v = state_vec

        elevator, ailerons, throttle = control

        # enforce velocity limits
        if v <= self.v_min and throttle < 0:
            throttle = 0
        elif v >= self.v_max and throttle > 0:
            throttle = 0

        # enforce roll limits
        if roll <= self.roll_min and ailerons < 0:
            ailerons = 0
        elif roll >= self.roll_max and ailerons > 0:
            ailerons = 0

        x_dot = v * math.cos(heading) * math.cos(gamma)
        y_dot = v * math.sin(heading) * math.cos(gamma)
        z_dot = -1 * v * math.sin(gamma)

        gamma_dot = elevator
        roll_dot = ailerons
        heading_dot = (self.g / v) * math.tan(roll)                      # g = 32.17 ft/s^2
        v_dot = throttle

        dx_vec = np.array([x_dot, y_dot, z_dot, heading_dot, gamma_dot, roll_dot, v_dot], dtype=np.float64)

        return dx_vec
