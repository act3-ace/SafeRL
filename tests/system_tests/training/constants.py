"""
This module hold constants and default values for training system tests.

Author: John McCarroll
"""


# results keys
CUSTOM_METRICS = "custom_metrics"
SUCCESS_MEAN = "outcome/success_mean"
TRAINING_ITERATIONS = "training_iteration"

# defaults
DEFAULT_GPUS = 0
DEFAULT_WORKERS = 6
DEFAULT_FAKE_GPUS = False
DEFAULT_OUTPUT = "test_data/training_output"

DEFAULT_SUCCESS_THRESHOLD = 0.9
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_SEED = 0
