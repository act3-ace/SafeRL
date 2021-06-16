import abc

import numpy as np


class Initializer(abc.ABC):
    def __init__(self, obj, init_config):
        self.env_obj = obj
        self.init_config = init_config

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError


class PassThroughInitializer(Initializer):
    def initialize(self):
        pass


class RandBoundsInitializer(Initializer):
    def initialize(self):
        if self.init_config is not None:
            new_params = {}
            for k, v in self.init_config.items():
                new_params[k] = v if type(v) != list else np.random.uniform(v[0], v[1])
            self.env_obj.reset(**new_params)
