"""
This module implements all fixtures common to testing the training functionality of our benchmarks.

Author: John McCarroll
"""

import pytest
import os
import shutil
import ray.rllib.agents.ppo as ppo
from ray import tune
from constants import *

from saferl.environment.callbacks import build_callbacks_caller, EpisodeOutcomeCallback, FailureCodeCallback,\
    RewardComponentsCallback
from saferl import lookup
from saferl.environment.utils import YAMLParser
from success_criteria import SuccessCriteria


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

    success_threshold = DEFAULT_SUCCESS_THRESHOLD
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

    # max_iterations = DEFAULT_MAX_ITERATIONS
    max_iterations = 5
    return max_iterations


test_configs = ["/home/john/AFRL/Dubins/have-deepsky/configs/rejoin/rejoin_default.yaml",
                "/home/john/AFRL/Dubins/have-deepsky/configs/docking/docking_default.yaml",
                "/home/john/AFRL/Dubins/have-deepsky/configs/rejoin/rejoin_3d_default.yaml",
                "/home/john/AFRL/Dubins/have-deepsky/configs/docking/docking_oriented_2d_default.yaml"]


# TODO: now, how to override if --configs param given?
# def pytest_addoption(parser):
#     parser.addoption(
#         "--configs", action="store", default="",
#         help="To override default benchmark config assay, assign the full path of a test configurations file."
#     )


def parse_config_file(test_config_path):
    config_paths = []
    with open(test_config_path, 'r') as file:
        for line in file:
            if line:
                # if line not empty
                if line[0] != "#":
                    # if line not comment
                    # add config path to test list
                    config_paths.append(line)

    return config_paths


@pytest.fixture()
def config_path(request):
    # retrieve path of test_config_file from command line
    test_config_file = request.config.getoption("--configs")
    if test_config_file and os.path.isfile(test_config_file):
        configs = parse_config_file(test_config_file)
        if configs:
            return configs

    # if no test_config_file given or if test_config_file empty, run default test assay
    return request.param


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
        config["num_gpus"] = DEFAULT_GPUS
        config["num_workers"] = DEFAULT_WORKERS
        config['_fake_gpus'] = DEFAULT_FAKE_GPUS
        config['seed'] = DEFAULT_SEED
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

    output_path = os.path.join(os.getcwd(), DEFAULT_OUTPUT)
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
    training_output : ExperimentAnalysis TODO: correct?
        A fixture that runs the experiment to completion and returns a dict of results.

    Returns
    -------
    success_rate : float
        The ratio of the successes to failures.
    """

    results = training_output.results[next(iter(training_output.results))]
    success_rate = results[CUSTOM_METRICS][SUCCESS_MEAN]
    return success_rate


@pytest.mark.system_test
@pytest.mark.parametrize("config_path", test_configs, indirect=True)
def test_training(success_rate, success_threshold):                 # add config_path to enable params?
    """
    This test ensures that an agent is still able to train on specified benchmarks. All benchmarks are tested by
    default.

    Parameters
    ----------
    success_rate : float
        The ratio of the successes to failures.
    success_threshold : float
        #TODO
    """

    assert success_rate >= success_threshold
