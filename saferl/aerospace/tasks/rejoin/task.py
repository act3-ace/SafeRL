import numpy as np
from saferl.environment import BaseEnv
from saferl.environment import RelativeCircle, RelativeCylinder
from saferl.aerospace.models import Dubins2dPlatform, Dubins3dPlatform


class DubinsRejoin(BaseEnv):

    def __init__(self, config):
        super(DubinsRejoin, self).__init__(config)
        self.timestep = 1

    def _setup_env_objs(self):
        if self.config['agent']['model'].lower() == '2d':
            wingman = Dubins2dPlatform(controller='agent', config=self.config['agent'])
            lead = Dubins2dPlatform()
        elif self.config['agent']['model'].lower() == '3d':
            wingman = Dubins3dPlatform(controller='agent', config=self.config['agent'])
            lead = Dubins3dPlatform()
        else:
            raise ValueError('Invalid agent type {} not supported'.format(self.config['agent']['model']))

        rejoin_region_type = self.config['rejoin_region']['type']
        if rejoin_region_type in ['circle', 'cylinder']:
            r_offset = self.config['rejoin_region']['range']
            radius = self.config['rejoin_region']['radius']

            if 'height' in self.config['rejoin_region']:
                height = self.config['rejoin_region']['height']
            else:
                height = 1

            aspect_angle = np.radians(self.config['rejoin_region']['aspect_angle'])
            if rejoin_region_type == 'circle':
                rejoin_region = RelativeCircle(lead, radius=radius, track_orientation=True, r_offset=r_offset,
                                               aspect_angle=aspect_angle)
            else:
                rejoin_region = RelativeCylinder(lead, radius=radius, height=height, track_orientation=True, r_offset=r_offset,
                                               aspect_angle=aspect_angle)

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
