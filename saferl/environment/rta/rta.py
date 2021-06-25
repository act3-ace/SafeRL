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

    def _filter_control(self, sim_state, step_size, control):
        self.monitor(sim_state, step_size, control)
        return self.generate_control(sim_state, step_size, control)

    def monitor(self, sim_state, step_size, control):
        self._monitor(sim_state, step_size, control)

    def generate_control(self, sim_state, step_size, control):
        return self._generate_control(sim_state, step_size, control)

    @abc.abstractmethod
    def _monitor(self, sim_state, step_size, control):
        raise NotImplementedError()

    @abc.abstractmethod
    def _generate_control(self, sim_state, step_size, control):
        raise NotImplementedError()


class SimplexModule(RTAModule):

    def __init__(self):
        self.rta_on = False
        super().__init__()

    def reset(self):
        self.rta_on = False
        super().reset()

    def _filter_control(self, sim_state, step_size, control):
        self.monitor(sim_state, step_size, control)
        if self.rta_on:
            return self.generate_control(sim_state, step_size, control)
        else:
            return control
