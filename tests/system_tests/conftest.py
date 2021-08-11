"""
This module defines fixtures common to the system_tests package.

Author: John McCarroll
"""

import pytest
import os
import ray.rllib.agents.ppo as ppo
from constants import *

from saferl.environment.callbacks import build_callbacks_caller, EpisodeOutcomeCallback, FailureCodeCallback,\
    RewardComponentsCallback

from saferl.environment.utils import YAMLParser, build_lookup, dict_merge


@pytest.fixture
def config(config_path, seed):
    """
    This fixture parses the yaml file found at the specified config_path and returns the resulting config dict.
    If the file does not exist, a FileNotFound error is raised.

    Parameters
    ----------
    config_path : str
        The full path of the desired experiment's config file.
    seed : int
        The seed passed to the environment to randomize training initialization.

    Returns
    -------
    config : dict
        A map of experiment config settings.
    """

    if os.path.exists(config_path):
        # Setup default PPO config
        config = ppo.DEFAULT_CONFIG.copy()
        config["num_gpus"] = DEFAULT_GPUS
        config["num_workers"] = DEFAULT_WORKERS
        config['_fake_gpus'] = DEFAULT_FAKE_GPUS
        config['seed'] = seed
        config['callbacks'] = build_callbacks_caller([EpisodeOutcomeCallback(),
                                                      FailureCodeCallback(),
                                                      RewardComponentsCallback()])

        # Setup custom config
        parser = YAMLParser(yaml_file=config_path, lookup=build_lookup())
        custom_config = parser.parse_env()
        config = dict_merge(config, custom_config, recursive=True)

        return config
    else:
        raise FileNotFoundError("Unable to locate: {}".format(config_path))


# fixtures for rllib environment setup and step testing
@pytest.fixture()
def environment(config):
    env = config["env"](config["env_config"])
    return env


@pytest.fixture()
def modified_environment_platform_state(environment, state, agent):
    if agent in environment.env_objs:
        for index, value in enumerate(state):
            environment.env_objs[agent].state._vector[index] = value
    return environment


@pytest.fixture()
def step(modified_environment_platform_state, action):
    obs, reward, done, info = modified_environment_platform_state.step(action)
    return obs, reward, done, info


@pytest.fixture()
def action():
    # placeholder to enable extensible constraint testing setup
    return None
