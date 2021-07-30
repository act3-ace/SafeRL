from saferl.environment.tasks.processor import RewardProcessor


class ConditionalRewardProcessor(RewardProcessor):
    def __init__(self, name, reward, cond_status):
        self.cond_status = cond_status
        self.last_step_size = 0
        super().__init__(name, reward)

    def reset(self, sim_state):
        super().reset(sim_state)
        self.last_step_size = 0

    def _increment(self, sim_state, step_size):
        self.last_step_size = step_size

    def _process(self, sim_state):
        cond = sim_state.status[self.cond_status]

        if cond:
            return self.reward
        else:
            return 0


class ProportionalRewardProcessor(RewardProcessor):
    def __init__(self, name, scale, bias, proportion_status, cond_status=None, cond_status_invert=False):
        self.scale = scale
        self.bias = bias
        self.proportion_status = proportion_status
        self.cond_status = cond_status
        self.cond_status_invert = cond_status_invert
        self.last_step_size = 0
        super().__init__(name, reward=0)

    def reset(self, sim_state):
        super().reset(sim_state)
        self.last_step_size = 0

    def _increment(self, sim_state, step_size):
        self.last_step_size = step_size

    def _process(self, sim_state):
        proportion = sim_state.status[self.proportion_status]
        if self.cond_status is None:
            cond = False
        else:
            cond = sim_state.status[self.cond_status]
            if self.cond_status_invert:
                cond = not cond

        if cond:
            reward = self.scale * proportion + self.bias
        else:
            reward = 0

        return reward
