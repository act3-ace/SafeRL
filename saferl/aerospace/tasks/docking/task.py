from saferl.environment.tasks.env import BaseEnv


class DockingEnv(BaseEnv):

    def __init__(self, config):
        super().__init__(config)

    def reset(self):
        return super().reset()

    def _step_sim(self, action):
        self.sim_state.env_objs['chief'].step(self.step_size)
        self.sim_state.env_objs['deputy'].step(self.step_size, action)

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
