"""
This module defines common fixtures for the constraints testing package.

Author: John McCarroll
"""

import pytest
# from tests.system_tests.constraints.constants import DEFAULT_SEED

# setting default here to avoid import issues
DEFAULT_SEED = 0

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


@pytest.fixture()
def seed():
    return DEFAULT_SEED
