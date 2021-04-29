from saferl.environment.tasks.env import BaseEnv


class DockingEnv(BaseEnv):

    def __init__(self, config):
        super(DockingEnv, self).__init__(config)
        self.step_size = 1

    def reset(self):
        return super(DockingEnv, self).reset()

    def _step_sim(self, action):
        self.sim_state.env_objs['chief'].pre_step(self.sim_state)
        self.sim_state.env_objs['deputy'].pre_step(self.sim_state, action)

        self.sim_state.env_objs['chief'].step(self.step_size)
        self.sim_state.env_objs['deputy'].step(self.step_size)

    def _generate_info(self):
        info = {
            'deputy': self.env_objs['deputy']._generate_info(),
            'chief': self.env_objs['chief']._generate_info(),
            'docking_region': self.env_objs['docking_region']._generate_info(),
            'failure': self.status['failure'],
            'success': self.status['success'],
            'status': self.status,
            'reward': self.reward_manager._generate_info(),
            'timestep_size': self.step_size
        }

        return info
