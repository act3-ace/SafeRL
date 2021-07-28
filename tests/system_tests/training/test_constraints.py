"""
This module tests the constraints of the aerospace environments.
"""

import pytest
import os
import shutil
import ray.rllib.agents.ppo as ppo
from ray import tune
from constants import *

from saferl.environment.callbacks import build_callbacks_caller, EpisodeOutcomeCallback, FailureCodeCallback,\
    RewardComponentsCallback

from saferl.environment.utils import YAMLParser, build_lookup, dict_merge
from success_criteria import SuccessCriteria


@pytest.fixture()
def base_environment(config):
    env = config["env"](config["env_config"])
    return env


@pytest.fixture()
def modified_environment(base_environment):
    # placeholder to enable extensible constraint testing setup
    return base_environment


@pytest.fixture()
def step(modified_environment, action=None):
    obs, reward, done, info = modified_environment.step(action)
    return obs, reward, done, info


class TestDockingVelocityConstraint:
    @pytest.fixture()
    def config_path(self):
        return "../configs/docking/docking_default.yaml"

    @pytest.fixture()
    def seed(self):
        return 0

    @pytest.fixture()
    def modified_environment(self, base_environment):
        base_environment.env_objs["deputy"].state._vector[0] = 0.1
        base_environment.env_objs["deputy"].state._vector[1] = 0.2
        base_environment.env_objs["deputy"].state._vector[2] = 5
        return base_environment

    @pytest.mark.system_test
    def test_velocity_constraint(self, step):
        obs, reward, done, info = step
        assert info["status"]["max_vel_constraint"]
        assert info["status"]["max_vel_limit"] > 1
        assert info["reward"]["max_vel_constraint"] < 0
        assert done


# problems:
# - new class for each test
# - config_path and seed need overwriting
# - lookup bug
# - no parameterization
