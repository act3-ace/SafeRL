from saferl.environment.tasks.env import BaseEnv


class DockingEnv(BaseEnv):

    def __init__(self, env_config):
        super().__init__(env_config)

    def reset(self):
        return super().reset()

    def _step_sim(self, action):
        self.sim_state.env_objs['chief'].step_compute(self.sim_state, self.step_size)
        self.sim_state.env_objs['deputy'].step_compute(self.sim_state, self.step_size, action)

        self.sim_state.env_objs['chief'].step_apply()
        self.sim_state.env_objs['deputy'].step_apply()

    def generate_info(self):
        info = {
            'deputy': self.env_objs['deputy'].generate_info(),
            'chief': self.env_objs['chief'].generate_info(),
            'docking_region': self.env_objs['docking_region'].generate_info(),
            'failure': self.status['failure'],
            'success': self.status['success'],
            'status': self.status,
            'reward': self.reward_manager.generate_info(),
            'timestep_size': self.step_size
        }

        return info
