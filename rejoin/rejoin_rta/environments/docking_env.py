from rejoin_rta.environments import BaseEnv
from rejoin_rta.aero_models.cwh_spacecraft import CWHSpacecraft
from rejoin_rta.utils.geometry import RelativeCircle2d, RelativeCylinder


class DockingEnv(BaseEnv):

    def __init__(self, config):
        super(DockingEnv, self).__init__(config)
        self.timestep = 1

    def _setup_env_objs(self):
        deputy = CWHSpacecraft(config=self.config['agent'])
        chief = CWHSpacecraft()

        if self.config['docking_region']['type'] == 'circle':
            radius = self.config['docking_region']['radius']
            docking_region = RelativeCircle2d(chief, radius=radius, x_offset=0, y_offset=0)
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
