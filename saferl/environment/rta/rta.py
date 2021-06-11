import abc


class RTAModule(abc.ABC):

    def __init__(self):
        self.enable = True

    def reset(self):
        self.enable = True

    def filter_control(self, sim_state, control):
        if self.enable:
            return self._filter_control(sim_state, control)
        else:
            return control

    def _filter_control(self, sim_state, control):
        self.monitor(sim_state, control)
        return self.generate_control(sim_state, control)

    def monitor(self, sim_state, control):
        self._monitor(sim_state, control)

    def generate_control(self, sim_state, control):
        return self._generate_control(sim_state, control)

    @abc.abstractmethod
    def _monitor(self, sim_state, control):
        raise NotImplementedError()

    @abc.abstractmethod
    def _generate_control(self, sim_state, control):
        raise NotImplementedError()


class SimplexModule(RTAModule):

    def __init__(self):
        self.rta_on = False
        super().__init__()

    def reset(self):
        self.rta_on = False
        super().reset()

    def _filter_control(self, sim_state, control):
        self.monitor(sim_state, control)
        if self.rta_on:
            return self.generate_control(sim_state, control)
        else:
            return control
