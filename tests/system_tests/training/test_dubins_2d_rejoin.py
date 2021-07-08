"""
This class holds all 2d dubins rejoin specific fixtures and tests
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

    config_path = "../../../configs/rejoin/rejoin_default.yaml"
    # TODO: remove brittle hardcoding (this requires pytest to be run from tests/ dir)
    return config_path


@pytest.fixture
def expr_name():
    """
    Returns
    -------
    name : str
        The name of the experiment.
    """

    name = "dubins_2d_rejoin_test"
    return name


@pytest.fixture
def stop_dict():
    """
    Returns
    -------
    stop_dict : dict
        A dict specifying stop criteria for the experiment.
    """

    stop_dict = {
        'training_iteration': 20
    }
    return stop_dict


@pytest.mark.system_test
def test_dubins_2d_rejoin(success_rate, success_threshold):
    """
    This test ensures that an agent is still able to train in the 2d rejoin environment.

    Parameters
    ----------
    success_rate : float
        The ratio of the successes to failures.
    """

    assert success_rate > success_threshold
