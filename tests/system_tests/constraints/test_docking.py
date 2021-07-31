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
        [1500, 0, 3.1, 0],
        [0, 1500, 0, 3.1],
        [1060.6, 1060.6, 0, 3.1],

        [0.1, 0, 0.1, 0],
        [0, 0.1, 0, 0.1],
        [0.1, 0.1, 0.1, 0.1],

        [500, 0, 1.1, 0],
        [0, 500, 0, 1.1],
        [353.6, 353.6, 0.73, 0.73]
    ]

    @pytest.fixture()
    def state(self, request):
        return request.param

    @pytest.fixture()
    def agent(self):
        return "deputy"

    @pytest.mark.system_test
    @pytest.mark.one
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
        [1500, 0, 2.9, 0],
        [0, 1500, 0, 2.9],
        [1060.6, 1060.6, 0, 3.1],

        [0.1, 0, 0.1, 0],
        [0, 0.1, 0, 0.1],
        [0.1, 0.1, 0.1, 0.1],

        [500, 0, 1.1, 0],
        [0, 500, 0, 1.1],
        [353.6, 353.6, 0.73, 0.73]
    ]

    @pytest.fixture()
    def state(self, request):
        return request.param

    @pytest.fixture()
    def agent(self):
        return "deputy"

    @pytest.mark.system_test
    @pytest.mark.one
    @pytest.mark.parametrize("state", test_states, indirect=True)
    def test_velocity_constraint(self, step):
        # decompose the results of an environment step and assert the desired response from the environment.
        obs, reward, done, info = step
        assert info["status"]["max_vel_constraint"]
        assert info["status"]["max_vel_limit"] > 1
        assert info["reward"]["components"]["step"]["max_vel_constraint"] < 0
        assert done
