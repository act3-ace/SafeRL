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


class CaseListInitializer(Initializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # start at -1 to account for initializer call in constructor
        self.iteration = -1
        self.sequential = self.init_config.get("sequential", False)
        self.case_list = self.init_config["case_list"]

        assert isinstance(self.sequential, bool), "sequntial must be a bool"
        assert isinstance(self.case_list, list), "case_list must be a list of dictionaries"


    def get_init_params(self):
        if self.sequential:
            case_idx = self.iteration % len(self.case_list)
        else:
            case_idx = np.random.randint(0, len(self.case_list))

        self.iteration += 1

        case = self.case_list[case_idx]

        return case

