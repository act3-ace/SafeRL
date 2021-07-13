import abc
import numpy as np


class RTAModule(abc.ABC):

    def __init__(self):
        self.platform = None

        self.enable = True
        self.intervening = False
        self.control_desired = None
        self.control_actual = None

        self.reset()

    def reset(self):
        self.enable = True
        self.intervening = False
        self.control_desired = None
        self.control_actual = None

    def setup(self, platform):
        self.platform = platform

    def filter_control(self, sim_state, step_size, control):
        self.control_desired = np.copy(control)

        if self.enable:
            self.control_actual = np.copy(self._filter_control(sim_state, step_size, control))
        else:
            self.control_actual = np.copy(control)

        return np.copy(self.control_actual)

    @abc.abstractmethod
    def _filter_control(self, sim_state, step_size, control):
        raise NotImplementedError()

    def generate_info(self):
        info = {
            'enable': self.enable,
            'intervening': self.intervening,
            'control_desired': self.control_desired,
            'control_actual': self.control_actual,
        }

        return info


class SimplexModule(RTAModule):

    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()

    def _filter_control(self, sim_state, step_size, control):
        self.monitor(sim_state, step_size, control)

        if self.intervening:
            return self.backup_control(sim_state, step_size, control)
        else:
            return control

    def monitor(self, sim_state, step_size, control):
        self.intervening = self._monitor(sim_state, step_size, control, self.intervening)

    def backup_control(self, sim_state, step_size, control):
        return self._backup_control(sim_state, step_size, control)

    @abc.abstractmethod
    def _monitor(self, sim_state, step_size, control, intervening):
        '''
        Returns
        -------
        bool
            True if unsafe
        '''
        raise NotImplementedError()

    @abc.abstractmethod
    def _backup_control(self, sim_state, step_size, control):
        raise NotImplementedError()
