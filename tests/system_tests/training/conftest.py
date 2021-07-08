"""
This module implements all fixtures common to testing the training functionality of our benchmarks.

Author: John McCarroll
"""

import pytest
import os
import shutil
import ray.rllib.agents.ppo as ppo
from ray import tune

from saferl.environment.callbacks import build_callbacks_caller, EpisodeOutcomeCallback, FailureCodeCallback,\
    RewardComponentsCallback, LoggingCallback, LogContents
from saferl import lookup
from saferl.environment.utils import YAMLParser
from success_criteria import SuccessCriteria


# define defaults
default_log_contents = (LogContents.INFO,)
default_gpus = 0
default_workers = 6
default_fake_gpus = False
default_seed = 100
default_output = "../../test_data/training_output"


@pytest.fixture()
def success_threshold():
    """
    This fixture determines the rate of successful episodes a task must reach during training in order to pass the
    system test.

    Returns
    -------
    success_threshold : float
        Desired rate of successful episodes to be confident task training functions appropriately.
    """

    success_threshold = 0.9
    return success_threshold


@pytest.fixture()
def max_iterations():
    """
    This fixture defines the maximum training iterations before termination during a test of task training
    functionality.

    Returns
    -------
    max_iters : int
        Maximum training iterations allowed before training termination.
    """

    max_iterations = 1000
    return max_iterations


@pytest.fixture
def config(config_path):
    """
    This fixture parses the yaml file found at the specified config_path and returns the resulting config dict.
    If the file does not exist, a FileNotFound error is raised.

    Parameters
    ----------
    config_path : str
        The full path of the desired experiment's config file.

    Returns
    -------
    config : dict
        A map of experiment config settings.
    """

    if os.path.exists(config_path):
        # Setup default PPO config
        config = ppo.DEFAULT_CONFIG.copy()
        config["num_gpus"] = default_gpus
        config["num_workers"] = default_workers
        config['_fake_gpus'] = default_fake_gpus
        config['seed'] = default_seed
        config['callbacks'] = build_callbacks_caller([EpisodeOutcomeCallback(),
                                                      FailureCodeCallback(),
                                                      RewardComponentsCallback()])

        # Setup custom config
        parser = YAMLParser(yaml_file=config_path, lookup=lookup)
        env, env_config = parser.parse_env()
        config['env'] = env
        config['env_config'] = env_config

        return config
    else:
        raise FileNotFoundError("Unable to locate: {}".format(config_path))


@pytest.fixture
def output_dir():
    """
    This fixture creates, returns, and tears down the output directory for test training logs, based on the
    default_output.

    Returns
    ----------
    output_path : str
        The full path of the desired experiment's config file.
    """

    output_path = os.path.join(os.getcwd(), default_output)
    os.makedirs(output_path, exist_ok=True)
    yield output_path
    shutil.rmtree(output_path)


@pytest.fixture
def training_output(config, output_dir, success_threshold, max_iterations):
    """
    This fixture runs training, populating the output_path with training logs.

    Parameters
    ----------
    config : dict
        A map of experiment config settings.
    output_dir : Fixture
        The full path of the desired experiment's config file.
    success_threshold : float
        Desired rate of successful episodes to be confident task training functions appropriately.
    max_iterations : int
        Maximum training iterations allowed before training termination.
    """

    results = tune.run(ppo.PPOTrainer,
                       config=config,
                       stop=SuccessCriteria(success_threshold=success_threshold, max_iterations=max_iterations),
                       local_dir=output_dir,
                       verbose=0)

    return results


@pytest.fixture
def success_rate(training_output):
    """
    This fixture parses a training log and returns the success ratio.

    Parameters
    ----------
    training_output : dict
        A fixture that runs the experiment to completion and returns a dict of results.

    Returns
    -------
    success_rate : float
        The ratio of the successes to failures.
    """

    results = training_output.results[next(iter(training_output.results))]
    success_rate = results["custom_metrics"]["outcome/success_mean"]
    return success_rate
