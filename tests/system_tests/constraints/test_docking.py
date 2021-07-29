"""
This module tests the constraints of the docking environment.

Author: Umberto Ravaioli John McCarroll
"""

import pytest


@pytest.fixture()
def config_path():
    # define the relative path of the environment under test in this module
    return "../configs/docking/docking_nominal.yaml"


class TestVelocityConstraint:
    # write one class per constraint under test

    @pytest.fixture()
    def modified_environment(self, base_environment):
        # override the modified_environment fixture to manually set up the state before "step()" is called on the
        # environment.
        base_environment.env_objs["deputy"].state._vector[0] = 0.1
        base_environment.env_objs["deputy"].state._vector[1] = 0.2
        base_environment.env_objs["deputy"].state._vector[2] = 5
        return base_environment

    @pytest.mark.system_test
    @pytest.mark.one
    def test_velocity_constraint(self, step):
        # decompose the results of an environment step and assert the desired response from the environment.
        obs, reward, done, info = step
        assert info["status"]["max_vel_constraint"]
        assert info["status"]["max_vel_limit"] > 1
        assert info["reward"]["max_vel_constraint"] < 0
        assert done

# - lookup bug
