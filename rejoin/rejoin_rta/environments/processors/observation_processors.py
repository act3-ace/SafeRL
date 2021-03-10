import numpy as np
from gym.spaces import Box

from rejoin.rejoin_rta.environments.processors import ObservationProcessor

# TODO: Implement Docking and Dubins observation processors


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
        obs = None
        if self.config['mode'] == '2d':
            obs = env_objs['deputy'].state2d
        elif self.config['mode'] == '3d':
            obs = env_objs['deputy'].state

        return obs
