"""
This module defines SuccessCriteria, a subclass of ray.tune.Stopper which is responsible for determining when a given
training run should be terminated.

Author: John McCarroll
"""

from ray.tune.stopper import Stopper
# from tests.system_tests.training.constants import CUSTOM_METRICS, SUCCESS_MEAN, TRAINING_ITERATIONS
from constants import *


class SuccessCriteria(Stopper):
    def __init__(self, success_threshold=DEFAULT_SUCCESS_THRESHOLD, max_iterations=DEFAULT_MAX_ITERATIONS):
        super().__init__()
        self.success_threshold = success_threshold
        self.max_iterations = max_iterations

    def __call__(self, trial_id, results):
        """
        This function checks if training episode success rate is above the acceptable threshold
        or if training has progressed longer than the maximum allowed iterations.
        It returns True, terminating the test if either condition is met.
        """

        terminate = False
        custom_metrics = results[CUSTOM_METRICS]

        if SUCCESS_MEAN in custom_metrics:
            if custom_metrics[SUCCESS_MEAN] >= self.success_threshold:
                terminate = True

        if results[TRAINING_ITERATIONS] > self.max_iterations:
            terminate = True

        return terminate

    def stop_all(self):
        pass
