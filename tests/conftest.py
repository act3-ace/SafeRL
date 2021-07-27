"""
This module implements all fixtures common to the entire test suite.

Author: John McCarroll
"""

import pytest
import os


def pytest_addoption(parser):
    """
    This fixture adds a custom command line option for running pytest. This custom option takes in the path of a system
    test config file, which lists paths to the environment (benchmark) configs desired to for testing. This enables easy
    testing of custom configs and creation of system test assays.

    Parameters
    ----------
    parser : pytest.Fixture
        The command line parser Fixture, provided by pytest.
    """

    parser.addoption(
        "--configs", action="store", default="",
        help="To override default benchmark config assay, assign the full path of a test configurations file."
    )


# def pytest_generate_tests(metafunc):
#     if 'player' in metafunc.fixturenames:
#         team_name = metafunc.config.getoption('team')
#
#         # you can move this part out to module scope if you want
#         with open('./teams.yml') as f:
#             teams = yaml.load(f)
#
#         metafunc.parametrize("player", teams.get(team_name, []))