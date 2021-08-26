"""
This module defines common fixtures for the constraints testing package.

Author: John McCarroll
"""

import pytest
# from tests.system_tests.constraints.constants import DEFAULT_SEED

# setting default here to avoid import issues
DEFAULT_SEED = 0


@pytest.fixture()
def seed():
    return DEFAULT_SEED
