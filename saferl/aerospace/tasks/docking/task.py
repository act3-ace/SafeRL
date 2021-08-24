from saferl.environment.tasks.env import BaseEnv
from saferl.environment.models.platforms import BasePlatform
import pdb

class DockingEnv(BaseEnv):

    def __init__(self, env_config):
        super().__init__(env_config)

    def reset(self):
        return super().reset()

    def _step_sim(self, action):
        agent_name = self.sim_state.agent.name
        for obj_name,obj in self.sim_state.env_objs.items():
            if isinstance(obj,BasePlatform):
                if obj_name == agent_name:
                    self.sim_state.env_objs[obj_name].step(self.step_size,action)
                else:
                    self.sim_state.env_objs[obj_name].step(self.step_size)

    def generate_info(self):

        info = {}
        for obj_name in self.env_objs:
            info[obj_name] = self.env_objs[obj_name].generate_info()

        for status in self.status:
            info[status] = self.status[status]

        info['status'] = self.status
        info['reward'] = self.reward_manager.generate_info()
        info['timestep_size'] = self.step_size

        return info
