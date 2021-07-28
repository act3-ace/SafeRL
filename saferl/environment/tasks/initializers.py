import abc

import numpy as np


class Initializer(abc.ABC):
    def __init__(self, obj, init_config):
        self.env_obj = obj
        self.init_config = init_config

    def initialize(self):
        if self.init_config is not None:
            new_params = self.get_init_params()
            self.env_obj.reset(**new_params)

    @abc.abstractmethod
    def get_init_params(self):
        raise NotImplementedError


class PassThroughInitializer(Initializer):
    def get_init_params(self):
        return self.init_config


class RandBoundsInitializer(Initializer):
    def get_init_params(self):
        new_params = {}
        for k, v in self.init_config.items():
            new_params[k] = v if type(v) != list else np.random.uniform(v[0], v[1])
        return new_params
