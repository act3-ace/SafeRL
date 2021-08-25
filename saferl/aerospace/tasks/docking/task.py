from saferl.environment.tasks.env import BaseEnv


class DockingEnv(BaseEnv):

    def __init__(self, env_config):
        super().__init__(env_config)

    def reset(self):
        return super().reset()
