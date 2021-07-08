"""
This class holds all 2d cwh docking specific fixtures and tests

Author: John McCarroll
"""

import pytest


@pytest.fixture
def config_path():
    """
    Returns
    -------
    config_path : str
        The path to the 2D Dubins Rejoin config file.
    """

    config_path = "../../../configs/docking/docking_default.yaml"
    # TODO: remove brittle hardcoding (this requires pytest to be run from tests/ dir)
    return config_path


@pytest.mark.system_test
def test_cwh_2d_docking(success_rate, success_threshold):
    """
    This test ensures that an agent is still able to train in the cwh 2d docking environment.

    Parameters
    ----------
    success_rate : float
        The ratio of the successes to failures.
    success_threshold : float
        #TODO
    """

    assert success_rate >= success_threshold
