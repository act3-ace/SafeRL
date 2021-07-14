"""
This class holds all training specific fixtures and tests.

Author: John McCarroll
"""

import pytest
from constants import *


"""
test_configs : Construct your Training Test Assay

Declare this variable as a list of tuples. Each tuple is a configuration for a training system test, containing three 
elements, in order: config_path, success_threshold, and max_iterations. 

config_path is the relative path to the config file for the training run under test. 
success_threshold is the decimal rate of successful episodes required to determine if training still functions.
max_iterations is the maximum allowed training iterations before test termination.
"""

# Define relative paths from the tests dir, where pytest should be run, to desired config files for trainings under
# test. Absolute paths will also run, for convenience.
REJOIN_DEFAULT_PATH = "../configs/rejoin/rejoin_default.yaml"
DOCKING_DEFAULT_PATH = "../configs/docking/docking_default.yaml"
REJOIN_3D_PATH = "../configs/rejoin/rejoin_3d_default.yaml"
DOCKING_ORIENTED_2D_PATH = "../configs/docking/docking_oriented_2d_default.yaml"

test_configs = [(REJOIN_DEFAULT_PATH, DEFAULT_SUCCESS_THRESHOLD, 200, DEFAULT_SEED),
                (DOCKING_DEFAULT_PATH,  DEFAULT_SUCCESS_THRESHOLD, 200, DEFAULT_SEED),
                (REJOIN_3D_PATH,  DEFAULT_SUCCESS_THRESHOLD, DEFAULT_MAX_ITERATIONS, DEFAULT_SEED),
                (DOCKING_ORIENTED_2D_PATH,  DEFAULT_SUCCESS_THRESHOLD, 500, DEFAULT_SEED)]


@pytest.mark.system_test
@pytest.mark.parametrize("config_path,success_threshold,max_iterations,seed", test_configs, indirect=True)
def test_training(success_rate, success_threshold):
    """
    This test ensures that an agent is still able to train on specified benchmarks. All benchmarks are tested by
    default.

    Parameters
    ----------
    success_rate : float
        The ratio of the successes to failures.
    success_threshold : float
        Desired rate of successful episodes to be confident task training functions appropriately.
    """

    assert success_rate >= success_threshold

