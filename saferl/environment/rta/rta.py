import abc


class RTAModule(abc.ABC):

    def __init__(self):
        self.rta_on = False

    def reset(self):
        self.rta_on = False

    def filter_control(self, sim_state, control):
        self.monitor(sim_state, control)
        return self.generate_control(sim_state, control)

    def monitor(self, sim_state, control):
        self._monitor(sim_state, control)

    def generate_control(self, sim_state, control):
        if self.rta_on:
            return self._generate_control(sim_state, control)
        else:
            return control

    @abc.abstractmethod
    def _monitor(self, sim_state, control):
        raise NotImplementedError()

    @abc.abstractmethod
    def _generate_control(self, sim_state, control):
        raise NotImplementedError()
