from saferl.environment.tasks.env import BaseEnv
from saferl.aerospace.models.dubins.rta import RTADubins2dCollision

class DubinsRejoin(BaseEnv):

    def __init__(self, config):
<<<<<<< HEAD
        super(DubinsRejoin, self).__init__(config)
        self.step_size = 0.1
        self.sim_state.env_objs['wingman'].rta_module = RTADubins2dCollision()
=======
        super().__init__(config)
        self.step_size = 1
>>>>>>> c49e6216869c2200e598a34f8e508c331e56cff6

    def reset(self):
        return super().reset()

    def _step_sim(self, action):

        self.sim_state.env_objs['lead'].pre_step(self.sim_state)
        self.sim_state.env_objs['wingman'].pre_step(self.sim_state, action)

        self.sim_state.env_objs['lead'].step(self.step_size)
        self.sim_state.env_objs['wingman'].step(self.step_size)

    def generate_info(self):
        info = {
            'wingman': self.env_objs['wingman'].generate_info(),
            'lead': self.env_objs['lead'].generate_info(),
            'rejoin_region': self.env_objs['rejoin_region'].generate_info(),
            'failure': self.status['failure'],
            'success': self.status['success'],
            'status': self.status,
<<<<<<< HEAD
            'reward': self.reward_manager._generate_info(),
            'timestep_size': self.step_size,
            'rta': self.env_objs['wingman'].rta_module.generate_info(),
=======
            'reward': self.reward_manager.generate_info(),
            'timestep_size': self.step_size
>>>>>>> c49e6216869c2200e598a34f8e508c331e56cff6
        }

        return info
