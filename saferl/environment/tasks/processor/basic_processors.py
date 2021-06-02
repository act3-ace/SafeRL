from saferl.environment.tasks.processor import StatusProcessor

# Is used a default failure processor
class TimeoutProcessor(StatusProcessor):
    def __init__(self, name='failure', timeout=1000):
        super().__init__(name=name)
        self.timeout = timeout

    def reset(self, sim_state):
        self.time_elapsed = 0

    def _increment(self, sim_state, step_size):
        # increment internal state
        self.time_elapsed += step_size

    def _process(self, sim_state):
        # process state and return status
        if self.time_elapsed > self.timeout:
            failure = 'timeout'
        else:
            failure = False

        return failure

# A status processor, can be used as a default success processors

class NeverSuccessProcessor(StatusProcessor):

    def __init__(self, name='success'):
        super().__init__(name=name)

    def reset(self, sim_state):
        pass

    def _increment(self, sim_state, step_size):
        # status derived directly from simulation state, therefore no state machine needed
        pass

    def _process(self, sim_state):
        # since a default , will always return false
        success = False
        return success
