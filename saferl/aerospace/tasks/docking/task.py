from saferl.environment.tasks.env import BaseEnv
from saferl.environment.utils import setup_env_objs_from_config


class DockingEnv(BaseEnv):

    def __init__(self, config):
        super().__init__(config)
        self.step_size = 1

    def _setup_env_objs(self):
        agent, env_objs = setup_env_objs_from_config(self.config)

        return agent, env_objs

    def reset(self):
        return super().reset()

    def _step_sim(self, action):
        self.env_objs['chief'].step(self.step_size)
        self.env_objs['deputy'].step(self.step_size, action)

    def generate_info(self):
        info = {
            'deputy': self.env_objs['deputy'].generate_info(),
            'chief': self.env_objs['chief'].generate_info(),
            'docking_region': self.env_objs['docking_region'].generate_info(),
            'failure': self.status_dict['failure'],
            'success': self.status_dict['success'],
            'status': self.status_dict,
            'reward': self.reward_manager.generate_info(),
            'timestep_size': self.step_size
        }

        return info
