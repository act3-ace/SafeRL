from aerospaceSafeRL.environment import BaseEnv
from aerospaceSafeRL.environment import RelativeCircle, RelativeCylinder
from aerospaceSafeRL.AerospaceModels import CWHSpacecraft2d, CWHSpacecraft3d


class DockingEnv(BaseEnv):

    def __init__(self, config):
        super(DockingEnv, self).__init__(config)
        self.timestep = 1

    def _setup_env_objs(self):
        if self.config['mode'].lower() == '2d':
            spacecraft_class = CWHSpacecraft2d
        elif self.config['mode'].lower() == '3d':
            spacecraft_class = CWHSpacecraft3d
        else:
            raise ValueError(
                "Unknown docking environment mode {}. Should be one of ['2d', '3d']".format(self.config['mode']))

        deputy = spacecraft_class(controller='agent', config=self.config['agent'])
        chief = spacecraft_class()

        if self.config['docking_region']['type'] == 'circle':
            radius = self.config['docking_region']['radius']
            docking_region = RelativeCircle(chief, radius=radius, x_offset=0, y_offset=0)
        elif self.config['docking_region']['type'] == 'cylinder':
            docking_region = RelativeCylinder(chief, x_offset=0, y_offset=0, z_offset=0,
                                              **self.config['docking_region']['params'])
        else:
            raise ValueError(
                'Invalid docking region type {} not supported'.format(self.config['docking_region']['type']))

        self.env_objs = {
            'deputy': deputy,
            'chief': chief,
            'docking_region': docking_region,
        }

        self.agent = deputy

    def reset(self):
        return super(DockingEnv, self).reset()

    def _step_sim(self, action):
        self.env_objs['chief'].step(self.timestep)
        self.env_objs['deputy'].step(self.timestep, action)

    def _generate_info(self):
        info = {
            'deputy': self.env_objs['deputy']._generate_info(),
            'chief': self.env_objs['chief']._generate_info(),
            'docking_region': self.env_objs['docking_region']._generate_info(),
            'failure': self.status_dict['failure'],
            'success': self.status_dict['success'],
            'status': self.status_dict,
            'reward': self.reward_manager._generate_info(),
        }

        return info
