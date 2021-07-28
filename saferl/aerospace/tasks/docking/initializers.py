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
        x_dot = self.init_config.get('x_dot', [-10, 10])
        y_dot = self.init_config.get('y_dot', [-10, 10])
        enforce_safety_constraints = self.init_config.get('enforce_safety_constraints', True)

        x_dot, y_dot = get_constrainted_velocity(ref, x, y, x_dot, y_dot,
                                                 enforce_safety_constraints=enforce_safety_constraints)

        new_params = {
            "x": x,
            "y": y,
            "x_dot": x_dot,
            "y_dot": y_dot
        }

        return new_params


def get_relative_rect_from_polar(ref, radius, angle):
    ref_x, ref_y = ref.x, ref.y
    x = ref_x + radius * np.cos(angle)
    y = ref_y + radius * np.sin(angle)
    return x, y


def get_constrainted_velocity(ref, x, y, x_dot_bounds, y_dot_bounds, mean_motion=0.001027, max_x_dot=10, max_y_dot=10,
                              enforce_safety_constraints=True):
    rel_x = x - ref.x
    rel_y = y - ref.y

    # Velocity bound based on relative position
    # TODO: get rid of hardcoded values in constraint
    max_vel = 0.2 + 2 * mean_motion * math.sqrt(rel_x**2 + rel_y**2)
    min_vel = -0.2 + 1/2 * mean_motion * math.sqrt(rel_x**2 + rel_y**2)

    # Clamp bounds on max value and velcity constraint
    if enforce_safety_constraints:
        x_dot_bounds = set_max_bounds(set_max_bounds(x_dot_bounds, max_x_dot), max_vel)
        y_dot_bounds = set_max_bounds(set_max_bounds(y_dot_bounds, max_y_dot), max_vel)

    # draw x_dot and y_dot values until constraint satisfied

    draw_attempts = 0
    constraint_satified = False
    while not constraint_satified:
        x_dot = draw_from_range(x_dot_bounds)
        y_dot = draw_from_range(y_dot_bounds)
        vel = math.sqrt(x_dot**2 + y_dot**2)
        if (not enforce_safety_constraints) or (max_vel > vel and min_vel < vel):
            constraint_satified = True

        draw_attempts += 1
        if draw_attempts >= 100:
            raise ValueError("Cannot find initialization that meets safety constraints")

    return x_dot, y_dot


def set_max_bounds(bounds, max_value):
    if isinstance(bounds, list):
        lower, upper = bounds[0], bounds[-1]
        upper = min(max_value, upper)
        lower = max(-max_value, lower)
        bounds = [lower, upper]
    else:
        bounds = min(bounds, max_value) if bounds > 0 else max(bounds, -max_value)
    return bounds


def draw_from_range(bounds):
    draw = bounds if type(bounds) != list else np.random.uniform(bounds[0], bounds[1])
    return draw
