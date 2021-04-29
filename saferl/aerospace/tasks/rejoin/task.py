from saferl.environment.tasks.env import BaseEnv
from saferl.aerospace.models.dubins.rta import RTADubins2dCollision

class DubinsRejoin(BaseEnv):

    def __init__(self, config):
        super(DubinsRejoin, self).__init__(config)
        self.step_size = 0.1
        self.sim_state.env_objs['wingman'].rta_module = RTADubins2dCollision()

    def reset(self):
        return super(DubinsRejoin, self).reset()

    def _step_sim(self, action):

        self.sim_state.env_objs['lead'].pre_step(self.sim_state)
        self.sim_state.env_objs['wingman'].pre_step(self.sim_state, action)

        self.sim_state.env_objs['lead'].step(self.step_size)
        self.sim_state.env_objs['wingman'].step(self.step_size)

    def _generate_info(self):
        info = {
            'wingman': self.env_objs['wingman']._generate_info(),
            'lead': self.env_objs['lead']._generate_info(),
            'rejoin_region': self.env_objs['rejoin_region']._generate_info(),
            'failure': self.status['failure'],
            'success': self.status['success'],
            'status': self.status,
            'reward': self.reward_manager._generate_info(),
            'timestep_size': self.step_size,
            'rta': self.env_objs['wingman'].rta_module.generate_info(),
        }

        return info
