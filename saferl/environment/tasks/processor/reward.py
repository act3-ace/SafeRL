import math

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
            cond = True
        else:
            cond = sim_state.status[self.cond_status]
            if self.cond_status_invert:
                cond = not cond

        if cond:
            reward = self.scale * proportion + self.bias
        else:
            reward = 0

        return reward


class DistanceExponentialChangeRewardProcessor(RewardProcessor):
    def __init__(self, name, c=2, initial_distance=150, agent=None, target=None):
        super().__init__(name, reward=0)
        self.agent = agent
        self.target = target
        self.a = math.log(2)/initial_distance
        self.c = c
        self.prev_dist = 0
        self.curr_dist = 0

    def reset(self, sim_state):
        self.prev_dist = 0
        self.curr_dist = 0

    def _increment(self, sim_state, step_size):
        # update distances
        self.prev_dist = self.curr_dist

        dist = sim_state.env_objs[self.target].position - sim_state.env_objs[self.agent].position
        self.curr_dist = np.linalg.norm(dist)

    def _process(self, sim_state):
        return self.c * (math.e ** (-self.a * self.curr_dist) - math.e ** (-self.a * self.prev_dist))
