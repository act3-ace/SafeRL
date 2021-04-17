from saferl.environment.tasks.env import BaseEnv
from saferl.environment.utils import setup_env_objs_from_config


class DubinsRejoin(BaseEnv):

    def __init__(self, config):
        super(DubinsRejoin, self).__init__(config)
        self.timestep = 1

    def _setup_env_objs(self):
        agent, env_objs = setup_env_objs_from_config(self.config)

        return agent, env_objs

    def reset(self):
        return super(DubinsRejoin, self).reset()

    def _step_sim(self, action):
        self.env_objs['lead'].step(self.timestep)
        self.env_objs['wingman'].step(self.timestep, action)

    def _generate_info(self):
        info = {
            'wingman': self.env_objs['wingman']._generate_info(),
            'lead': self.env_objs['lead']._generate_info(),
            'rejoin_region': self.env_objs['rejoin_region']._generate_info(),
            'failure': self.status_dict['failure'],
            'success': self.status_dict['success'],
            'status': self.status_dict,
            'reward': self.reward_manager._generate_info(),
            'timestep_size': self.timestep
        }

        return info
