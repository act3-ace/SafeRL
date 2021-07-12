"""
This class holds all training specific fixtures and tests.

Author: John McCarroll
"""

import pytest
import os


def parse_config_file(test_config_path):
    config_paths = []
    with open(test_config_path, 'r') as file:
        for line in file:
            if line[0] != "#" and line[0] != '\n':
                # if line not comment or empty, add config path to test list
                config_paths.append(line.strip())

    return config_paths


# @pytest.fixture()
# def test_assay(request):
#     # retrieve path of test_config_file from command line
#     test_config_file = request.config.getoption("--configs")
#
#     if test_config_file and os.path.isfile(test_config_file):
#         configs = parse_config_file(test_config_file)
#         if configs:
#             return configs
#
#     # if no test_config_file given or if test_config_file empty, run default test assay
#     return default_test_configs


default_test_configs = ["/home/john/AFRL/Dubins/have-deepsky/configs/rejoin/rejoin_default.yaml",
                        "/home/john/AFRL/Dubins/have-deepsky/configs/docking/docking_default.yaml",
                        "/home/john/AFRL/Dubins/have-deepsky/configs/rejoin/rejoin_3d_default.yaml",
                        "/home/john/AFRL/Dubins/have-deepsky/configs/docking/docking_oriented_2d_default.yaml"]


@pytest.mark.system_test
@pytest.mark.parametrize("config_path", default_test_configs, indirect=True)
def test_training(success_rate, success_threshold):
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

