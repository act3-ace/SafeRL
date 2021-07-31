"""
This module tests the constraints of the docking environment.

Author: Umberto Ravaioli John McCarroll
"""

import pytest


@pytest.fixture()
def config_path():
    # define the relative path of the environment under test in this module
    return "../configs/docking/docking_nominal.yaml"


class TestVelocityConstraintExceedance:
    # write one class per constraint under test
    # testing exceedance of velocity limits

    test_states = [
        [1500, 0, 3.3, 0],
        [0, 1500, 0, 3.3],
        [1060.6, 1060.6, 2.4, 2.4],

        [0.21, 0, 0.1, 0],
        [0, 0.21, 0, 0.1],
        [0.15, 0.15, 0.075, 0.075],

        [500, 0, 1.25, 0],
        [0, 500, 0, 1.25],
        [353.6, 353.6, 0.89, 0.89]
    ]

    @pytest.fixture()
    def state(self, request):
        return request.param

    @pytest.fixture()
    def agent(self):
        return "deputy"

    @pytest.mark.system_test
    @pytest.mark.parametrize("state", test_states, indirect=True)
    def test_velocity_constraint(self, step):
        # decompose the results of an environment step and assert the desired response from the environment.
        obs, reward, done, info = step
        assert info["status"]["max_vel_constraint"]
        assert info["status"]["max_vel_limit"] > 1
        assert info["reward"]["components"]["step"]["max_vel_constraint"] < 0
        assert done


class TestVelocityConstraintConformity:
    # write one class per constraint under test
    # testing conforming to velocity limits

    test_states = [
        [1500, 0, 3.2, 0],
        [0, 1500, 0, 3.2],
        [1060.6, 1060.6, 2.3, 2.3],

        [0.21, 0, 0.09, 0],
        [0, 0.21, 0, 0.09],
        [0.15, 0.15, 0.07, 0.07],

        [500, 0, 1.2, 0],
        [0, 500, 0, 1.2],
        [353.6, 353.6, 0.84, 0.84]
    ]

    @pytest.fixture()
    def state(self, request):
        return request.param

    @pytest.fixture()
    def agent(self):
        return "deputy"

    @pytest.mark.system_test
    @pytest.mark.parametrize("state", test_states, indirect=True)
    def test_velocity_constraint(self, step):
        # decompose the results of an environment step and assert the desired response from the environment.
        obs, reward, done, info = step
        assert not info["status"]["max_vel_constraint"]
        assert info["status"]["max_vel_limit"] < 1
        assert info["reward"]["components"]["step"]["max_vel_constraint"] > 0
        assert not done


# TODO: test failure / success conditions IN docking region
# TODO: can test values come from config? or always hard code...
# TODO: file not found bug
