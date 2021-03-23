import numpy as np

from aerospaceSafeRL.environment import BaseEnv
from aerospaceSafeRL.environment import RelativeCircle
from aerospaceSafeRL.AerospaceModels import Dubins2dPlatform


class DubinsRejoin(BaseEnv):

    def __init__(self, config):
        super(DubinsRejoin, self).__init__(config)
        self.timestep = 1

    def _setup_env_objs(self):
        wingman = Dubins2dPlatform(controller='agent', config=self.config['agent'])
        lead = Dubins2dPlatform()

        if self.config['rejoin_region']['type'] == 'circle':
            r_offset = self.config['rejoin_region']['range']
            radius = self.config['rejoin_region']['radius']
            aspect_angle = np.radians(self.config['rejoin_region']['aspect_angle'])
            rejoin_region = RelativeCircle(lead, radius=radius, track_orientation=True, r_offset=r_offset, aspect_angle=aspect_angle)
        else:
            raise ValueError('Invalid rejoin region type {} not supported'.format(self.config['rejoin_region']['type']))

        self.env_objs = {
            'wingman': wingman,
            'lead': lead,
            'rejoin_region': rejoin_region,
        }

        self.agent = wingman

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
