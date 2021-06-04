import abc

import numpy as np


class Initializer(abc.ABC):
    def __init__(self, env_obj):
        self.env_obj = env_obj
        self.init_configs = {name: obj.state.init_params for name, obj in env_objs.items()}

    @abc.abstractmethod
    def initialize(self):
        raise NotImplementedError


class PassThroughInitializer(Initializer):
    def initialize(self):
        pass


class RandBoundsInitializer(Initializer):
    def initialize(self):
        for name, obj in self.env_objs.items():
            cfg = self.init_configs[name]
            assert cfg.keys() == obj.state.init_params.keys()
            for k, v in cfg:
                obj.state.init_params[k] = v if type(v) != list else np.random.uniform(v[0], v[1])
