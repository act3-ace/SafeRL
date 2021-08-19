from saferl.environment.tasks.env import BaseEnv
from saferl.aerospace.tasks.docking.render import DockingRender


class DockingEnv(BaseEnv):

    def __init__(self, env_config):
        super().__init__(env_config)
        render_config = env_config["render"] if "render" in env_config.keys() else {}
        self.renderer = DockingRender(**render_config)

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

    def render(self, mode='human'):
        self.renderer.renderSim(state=self.sim_state)
