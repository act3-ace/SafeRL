"""
This module defines SuccessCriteria, a subclass of ray.tune.Stopper which is responsible for determining when a given
training run should be terminated.

Author: John McCarroll
"""

from ray.tune.stopper import Stopper


class SuccessCriteria(Stopper):
    def __init__(self, success_threshold=0.9, max_iterations=1000):
        super().__init__()
        self.success_threshold = success_threshold
        self.max_iterations = max_iterations

    def __call__(self, trial_id, results):
        if results["custom_metrics"]["outcome/success_mean"] >= self.success_threshold \
                or results["training_iteration"] >= self.max_iterations:
            return True
        else:
            return False

    def stop_all(self):
        pass
