from saferl.environment.tasks.env import BaseEnv


class DubinsRejoin(BaseEnv):

    def __init__(self, config):
        super(DubinsRejoin, self).__init__(config)
        self.step_size = 1

    def reset(self):
        return super(DubinsRejoin, self).reset()

    def _step_sim(self, action):
        self.env_objs['lead'].step(self.step_size)
        self.env_objs['wingman'].step(self.step_size, action)

    def _generate_info(self):
        info = {
            'wingman': self.env_objs['wingman']._generate_info(),
            'lead': self.env_objs['lead']._generate_info(),
            'rejoin_region': self.env_objs['rejoin_region']._generate_info(),
            'failure': self.status_dict['failure'],
            'success': self.status_dict['success'],
            'status': self.status_dict,
            'reward': self.reward_manager._generate_info(),
            'timestep_size': self.step_size
        }

        return info
