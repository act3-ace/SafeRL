import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from rejoin_rta.environments.processors import ObservationProcessor


class DockingObservationProcessor(ObservationProcessor):
    def __init__(self, config, name="docking_observation"):
        super().__init__(config=config, name=name)

        low = np.finfo(np.float32).min
        high = np.finfo(np.float32).max

        if self.config['mode'] == '2d':
            self.observation_space = Box(low=low, high=high, shape=(4,))
        elif self.config['mode'] == '3d':
            self.observation_space = Box(low=low, high=high, shape=(6,))
        else:
            raise ValueError("Invalid observation mode {}. Should be one of ".format(self.config['mode']))

    def generate_observation(self, env_objs):
        obs = env_objs['deputy'].state.vector
        return obs


class DubinsObservationProcessor(ObservationProcessor):
    def __init__(self, config):
        super().__init__(config=config, name="dubins_observation")

        if self.config['mode'] == 'rect':
            self.observation_space = Box(low=-1, high=1, shape=(8,))
            self.obs_norm_const = np.array([10000, 10000, 10000, 10000, 100, 100, 100, 100], dtype=np.float64)

        elif self.config['mode'] == 'magnorm':
            self.observation_space = Box(low=-1, high=1, shape=(12,))
            self.obs_norm_const = np.array([10000, 1, 1, 10000, 1, 1, 100, 1, 1, 100, 1, 1], dtype=np.float64)

    def generate_observation(self, env_objs):
        def vec2magnorm(vec):
            norm = np.linalg.norm(vec)
            mag_norm_vec = np.concatenate( ( [norm], vec / norm ) )
            return mag_norm_vec

        wingman_lead_r = env_objs['lead'].position - env_objs['wingman'].position
        wingman_rejoin_r = env_objs['rejoin_region'].position - env_objs['wingman'].position

        wingman_vel = env_objs['wingman'].velocity
        lead_vel = env_objs['lead'].velocity

        reference_rotation = Rotation.from_quat([0, 0, 0, 1])
        if self.config['reference'] == 'wingman':
            reference_rotation = env_objs['wingman'].orientation.inv()

        wingman_lead_r = reference_rotation.apply(wingman_lead_r)
        wingman_rejoin_r = reference_rotation.apply(wingman_rejoin_r)

        wingman_vel = reference_rotation.apply(wingman_vel)
        lead_vel = reference_rotation.apply(lead_vel)

        if self.config['mode'] == 'magnorm':
            wingman_lead_r = vec2magnorm(wingman_lead_r)
            wingman_rejoin_r = vec2magnorm(wingman_rejoin_r)

            wingman_vel = vec2magnorm(wingman_vel)
            lead_vel = vec2magnorm(lead_vel)

        obs = np.concatenate( [
            wingman_lead_r[0:3],
            wingman_rejoin_r[0:3],
            wingman_vel[0:3],
            lead_vel[0:3],
        ] )

        # normalize observation
        obs = np.divide(obs, self.obs_norm_const)

        obs = np.clip(obs, -1, 1)

        return obs
