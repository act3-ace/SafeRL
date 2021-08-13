"""
This module tests the constraints of the docking environment.

Author: Umberto Ravaioli and John McCarroll
"""

import pytest


@pytest.fixture()
def config_path():
    # define the relative path of the environment under test in this module
    return "../configs/docking/docking_nominal.yaml"


@pytest.fixture()
def agent():
    return "deputy"


@pytest.fixture()
def expected_max_vel_limit(request):
    return request.param


class TestVelocityConstraintExceedance:
    # testing exceedance of velocity limits

    test_states = [
        ([1500, 0, 3.3, 0], 3.288),
        ([0, 1500, 0, 3.3], 3.288),
        ([1060.6, 1060.6, 2.4, 2.4], 3.288),

        ([0.11, 0, 0.21, 0], 0.200),
        ([0, 0.11, 0, 0.21], 0.200),
        ([0.077, 0.077, 0.149, 0.149], 0.200),

        ([500, 0, 1.25, 0], 1.229),
        ([0, 500, 0, 1.25], 1.229),
        ([353.6, 353.6, 0.89, 0.89], 1.230)
    ]

    @pytest.fixture()
    def state(self, request):
        return request.param

    @pytest.mark.system_test
    @pytest.mark.parametrize("state,expected_max_vel_limit", test_states, indirect=True)
    def test_velocity_constraint(self, step, expected_max_vel_limit):
        # decompose the results of an environment step and assert the desired response from the environment.
        obs, reward, done, info = step
        assert not info["status"]["max_vel_constraint"]
        assert round(info["status"]["max_vel_limit"], 3) == expected_max_vel_limit
        assert info["reward"]["components"]["step"]["max_vel_constraint"] < 0
        assert done


class TestVelocityConstraintConformity:
    # testing conforming to velocity limits

    test_states = [
        # below boundary
        ([1500, 0, 3.2, 0], 3.287),
        ([0, 1500, 0, 3.2], 3.287),
        ([1060.6, 1060.6, 2.3, 2.3], 3.287),

        ([0.11, 0, 0.19, 0], 0.200),
        ([0, 0.11, 0, 0.19], 0.200),
        ([0.077, 0.077, 0.134, 0.134], 0.200),

        ([500, 0, 1.2, 0], 1.229),
        ([0, 500, 0, 1.2], 1.229),
        ([353.6, 353.6, 0.84, 0.84], 1.229),

        # on boundary
        ([1500, 0, 3.287, 0], 3.288),# info[status][max_vel_constraint] == False
        ([0, 1500, 0, 3.287], 3.288),

        ([0.11, 0, 0.2, 0], 0.200),
        ([0, 0.11, 0, 0.2], 0.200),

        ([500, 0, 1.229, 0], 1.229),# info[status][max_vel_constraint] == False
        ([0, 500, 0, 1.229], 1.229)

    ]

    @pytest.fixture()
    def state(self, request):
        return request.param

    @pytest.mark.system_test
    @pytest.mark.parametrize("state,expected_max_vel_limit", test_states, indirect=True)
    def test_velocity_constraint(self, step, expected_max_vel_limit):
        # decompose the results of an environment step and assert the desired response from the environment.
        obs, reward, done, info = step
        assert info["status"]["max_vel_constraint"]
        assert round(info["status"]["max_vel_limit"], 3) == expected_max_vel_limit
        assert info["reward"]["components"]["step"]["max_vel_constraint"] == 0
        assert not done


class TestDockingVelocityConstraintExceedance:
    # testing exceedance of velocity limits inside docking region (ie. crashing)

    @pytest.fixture()
    def state(self, request):
        return [0.5, 0, 0.21, 0]

    @pytest.mark.system_test
    def test_velocity_constraint(self, step):
        # decompose the results of an environment step and assert the desired response from the environment.
        obs, reward, done, info = step
        assert not info["status"]["max_vel_constraint"]
        # assert round(info["status"]["max_vel_limit"], 3) == expected_max_vel_limit
        assert not info["status"]["success"]
        assert info["status"]["failure"] == "crash"


class TestDockingVelocityConstraintConformity:
    # testing conforming to velocity limits inside docking region (ie. successful docking)

    test_states = [
        [0.5, 0, 0.19, 0],
        [0.5, 0, 0.2, 0]
    ]

    @pytest.fixture()
    def state(self, request):
        return request.param

    @pytest.mark.system_test
    @pytest.mark.parametrize("state", test_states, indirect=True)
    def test_velocity_constraint(self, step):
        # decompose the results of an environment step and assert the desired response from the environment.
        obs, reward, done, info = step
        assert info["status"]["max_vel_constraint"]
        # assert info["status"]["max_vel_limit"] == expected_max_vel_limit
        assert info["status"]["success"]
        assert not info["status"]["failure"]
