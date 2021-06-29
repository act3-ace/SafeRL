import abc


class RTAModule(abc.ABC):

    def __init__(self):
        self.platform = None

        self.enable = True

    def reset(self):
        self.enable = True

    def setup(self, platform):
        self.platform = platform

    def filter_control(self, sim_state, step_size, control):
        if self.enable:
            return self._filter_control(sim_state, step_size, control)
        else:
            return control

    @abc.abstractmethod
    def _filter_control(self, sim_state, step_size, control):
        raise NotImplementedError()

    def generate_info(self):
        info = {
            'enable': self.enable,
        }

        return info


class SimplexModule(RTAModule):

    def __init__(self):
        self.backup_on = False
        super().__init__()

    def reset(self):
        self.backup_on = False
        super().reset()

    def _filter_control(self, sim_state, step_size, control):
        self.backup_on = self.monitor(sim_state, step_size, control)

        if self.backup_on:
            return self.backup_control(sim_state, step_size, control)
        else:
            return control

    def monitor(self, sim_state, step_size, control):
        return self._monitor(sim_state, step_size, control)

    def backup_control(self, sim_state, step_size, control):
        return self._backup_control(sim_state, step_size, control)

    @abc.abstractmethod
    def _monitor(self, sim_state, step_size, control):
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

    def generate_info(self):
        info = {
            'backup_on': self.backup_on,
        }

        info_parent = super().generate_info()
        info_ret = {**info_parent, **info}

        return info_ret
