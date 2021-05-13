from saferl.environment.tasks.env import BaseEnv


class DubinsRejoin(BaseEnv):

    def __init__(self, config):
        super().__init__(config)

    def reset(self):
        return super().reset()

    def _step_sim(self, action):
        self.sim_state.env_objs['lead'].step(self.step_size)
        self.sim_state.env_objs['wingman'].step(self.step_size, action)

    def generate_info(self):
        info = {
            'wingman': self.env_objs['wingman'].generate_info(),
            'lead': self.env_objs['lead'].generate_info(),
            'rejoin_region': self.env_objs['rejoin_region'].generate_info(),
            'failure': self.status['failure'],
            'success': self.status['success'],
            'status': self.status,
            'reward': self.reward_manager.generate_info(),
            'timestep_size': self.step_size
        }

        return info
