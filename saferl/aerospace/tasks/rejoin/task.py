from saferl.environment.tasks.env import BaseEnv


class DubinsRejoin(BaseEnv):

    def __init__(self, env_config):
        super().__init__(env_config)
        if "render" in env_config.keys():
            render_config = env_config["render"]
            # get safety margin dims
            safety_margin = None
            for proc in self.status_manager.processors:
                if hasattr(proc, "safety_margin"):
                    safety_margin = proc.safety_margin['aircraft']

            from saferl.aerospace.tasks.rejoin.render import RejoinRenderer
            self.renderer = RejoinRenderer(**render_config, safety_margin=safety_margin)

    def reset(self):
        return super().reset()
