import math
import numpy as np

from saferl.environment.tasks.initializers import Initializer


class ConstrainedDeputyPolarInitializer(Initializer):
    def get_init_params(self):
        # Get reference
        ref = self.init_config["ref"]

        # Get radius
        radius = self.init_config["radius"]
        radius = radius if type(radius) != list else np.random.uniform(radius[0], radius[1])

        # Get angle
        angle = self.init_config["angle"]
        angle = angle if type(angle) != list else np.random.uniform(angle[0], angle[1])

        x, y = get_relative_rect_from_polar(ref, radius, angle)

        # Get constrained velocity
        x_dot, y_dot = self.init_config["x_dot"], self.init_config["y_dot"]
        x_dot, y_dot = get_constrainted_velocity(
            reference=ref,
            x=x,
            y=y,
            x_dot=x_dot,
            y_dot=y_dot,
        )

        new_params = {
            "x": x,
            "y": y,
            "x_dot": x_dot,
            "y_dot": y_dot
        }
        for k, v in self.init_config.items():
            if k in new_params:
                v = new_params[k]
            new_params[k] = v if type(v) != list else np.random.uniform(v[0], v[1])
        return new_params


def get_relative_rect_from_polar(reference, radius, angle):
    ref_x, ref_y = reference.x, reference.y
    x = ref_x + radius * np.cos(angle)
    y = ref_y + radius * np.sin(angle)
    return x, y


def get_constrainted_velocity(reference, x, y, x_dot, y_dot, mean_motion=0.001027, max_x_dot=10, max_y_dot=10):
    def set_max_bounds(bounds, max_value):
        if isinstance(bounds, list):
            lower, upper = bounds[0], bounds[-1]
            upper = min(max_value, upper)
            lower = max(-max_value, lower)
            bounds[0] = lower
            bounds[-1] = upper
        else:
            bounds = min(bounds, max_value) if bounds > 0 else max(bounds, -max_value)
        return bounds

    def max_abs_value(bounds):
        if isinstance(bounds, list):
            if abs(bounds[0]) >= abs(bounds[-1]):
                max_abs_bounds = bounds[0]
                index = 0
            else:
                max_abs_bounds = bounds[-1]
                index = -1
        else:
            max_abs_bounds = bounds
            index = None
        return max_abs_bounds, index

    def reduce_components(x_dot, y_dot, constraint):
        max_abs_x_dot, max_abs_x_dot_idx = max_abs_value(x_dot)
        max_abs_y_dot, max_abs_y_dot_idx = max_abs_value(y_dot)
        if abs(max_abs_x_dot) > constraint and abs(max_abs_y_dot) > constraint:
            # Reduce both components
            new_value = math.sqrt((constraint ** 2) / 2)
            x_dot_value = new_value if max_abs_x_dot >= 0 else -new_value
            y_dot_value = new_value if max_abs_y_dot >= 0 else -new_value
        elif abs(max_abs_x_dot) >= abs(max_abs_y_dot):
            # abs(x_dot) > abs(y_dot). Reduce x_dot.
            new_value = math.sqrt(constraint ** 2 - max_abs_y_dot ** 2)
            x_dot_value = new_value if max_abs_x_dot >= 0 else -new_value
            y_dot_value = max_abs_y_dot
        else:
            # Reduce y_dot
            new_value = math.sqrt(constraint ** 2 - max_abs_x_dot ** 2)
            y_dot_value = new_value if max_abs_y_dot >= 0 else -new_value
            x_dot_value = max_abs_x_dot

        # Assign new values
        if max_abs_x_dot_idx is not None:
            x_dot[max_abs_x_dot_idx] = x_dot_value
        else:
            x_dot = x_dot_value
        if max_abs_y_dot_idx is not None:
            y_dot[max_abs_y_dot_idx] = y_dot_value
        else:
            y_dot = y_dot_value
        return x_dot, y_dot

    rel_x = x - reference.x
    rel_y = y - reference.y

    # Velocity bound based on relative position
    # TODO: get rid of hardcoded values in constraint
    constraint = 0.2 + 2 * mean_motion * math.sqrt(rel_x**2 + rel_y**2)

    # Clamp bounds on max value
    x_dot = set_max_bounds(x_dot, max_x_dot)
    y_dot = set_max_bounds(y_dot, max_y_dot)

    # Check constraint on largest possible velocity
    max_abs_x_dot, max_abs_x_dot_idx = max_abs_value(x_dot)
    max_abs_y_dot, max_abs_y_dot_idx = max_abs_value(y_dot)
    while math.sqrt(max_abs_x_dot ** 2 + max_abs_y_dot ** 2) > constraint:
        x_dot, y_dot = reduce_components(x_dot, y_dot, constraint)
        max_abs_x_dot, max_abs_x_dot_idx = max_abs_value(x_dot)
        max_abs_y_dot, max_abs_y_dot_idx = max_abs_value(y_dot)

    return x_dot, y_dot
