from saferl.environment.tasks.env import BaseEnv
from saferl.aerospace.tasks.rejoin.render import RejoinRenderer


class DubinsRejoin(BaseEnv):

    def __init__(self, env_config):
        super().__init__(env_config)
        render_config = env_config["render"] if "render" in env_config.keys() else {}
        # get safety margin dims
        safety_margin = None
        for proc in self.status_manager.processors:
            if hasattr(proc, "safety_margin"):
                safety_margin = proc.safety_margin['aircraft']

        self.renderer = RejoinRenderer(**render_config, safety_margin=safety_margin)

    def reset(self):
        return super().reset()
