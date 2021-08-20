"""
This module implements all fixtures common to testing the training functionality of our benchmarks.

Author: John McCarroll
"""

import pytest
import os
import shutil
import ray.rllib.agents.ppo as ppo
from ray import tune
from constants import DEFAULT_OUTPUT, CUSTOM_METRICS, SUCCESS_MEAN

from success_criteria import SuccessCriteria


@pytest.fixture()
def success_threshold(request):
    """
    This fixture determines the rate of successful episodes a task must reach during training in order to pass the
    system test. The returned success_threshold float is parameterized to return value(s) defined in test_configs
    variable in the test_training.py module.

    Parameters
    ----------
    request : Fixture
        A built-in pytest Fixture used to provide info on the executing test function.

    Returns
    -------
    success_threshold : float
        Desired rate of successful episodes to be confident task training functions appropriately.
    """

    return request.param


@pytest.fixture()
def max_iterations(request):
    """
    This fixture defines the maximum training iterations before termination during a test of task training
    functionality. The returned max_iterations int is parameterized to return value(s) defined in test_configs variable
    in the test_training.py module.

    Parameters
    ----------
    request : Fixture
        A built-in pytest Fixture used to provide info on the executing test function.

    Returns
    -------
    max_iterations : int
        Maximum training iterations allowed before training termination.
    """

    return request.param


@pytest.fixture()
def config_path(request):
    """
    This fixture defines the relative path to the YAML config file which defines the training under test. The returned
    config_path string is parameterized to return value(s) defined in test_configs variable in the test_training.py
    module.

    Parameters
    ----------
    request : Fixture
        A built-in pytest Fixture used to provide info on the executing test function.

    Returns
    -------
    config_path : str
        Relative path to the training config file.
    """
    return request.param


@pytest.fixture()
def seed(request):
    """
    This fixture defines the relative path to the YAML config file which defines the training under test. The returned
    config_path string is parameterized to return value(s) defined in test_configs variable in the test_training.py
    module.

    Parameters
    ----------
    request : Fixture
        A built-in pytest Fixture used to provide info on the executing test function.

    Returns
    -------
    seed : int
        The seed passed to the environment to randomize training initialization.
    """
    return request.param


@pytest.fixture
def output_dir():
    """
    This fixture creates, returns, and tears down the output directory for test training logs, based on the
    default_output.

    *Note: Due to the teardown of the entire output dir after each test, do NOT run multiple instances of pytest
    concurrently on the same machine! This will cause errors where test outputs are deleted midway through a test.

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
    training_output : ray.tune.analysis.experiment_analysis.ExperimentAnalysis
        A fixture that runs the experiment to completion and returns the results.

    Returns
    -------
    success_rate : float
        The ratio of the successes to failures.
    """

    results = training_output.results[next(iter(training_output.results))]
    success_rate = results[CUSTOM_METRICS][SUCCESS_MEAN]
    return success_rate
