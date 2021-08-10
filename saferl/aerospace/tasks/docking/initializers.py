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

        x_dot, y_dot = get_constrainted_velocity(ref, x, y)

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


def get_constrainted_velocity(ref, x, y, mean_motion=0.001027, max_vel_constraint=10):
    rel_x = x - ref.x
    rel_y = y - ref.y

    # Velocity bound based on relative position
    # TODO: get rid of hardcoded values in constraint
    park_orbit_max_vel = 0.2 + 2 * mean_motion * math.sqrt(rel_x**2 + rel_y**2)
    park_orbit_min_vel = -0.2 + 1/2 * mean_motion * math.sqrt(rel_x**2 + rel_y**2)

    max_vel = min(park_orbit_max_vel, max_vel_constraint)
    min_vel = park_orbit_min_vel

    speed = draw_from_range([min_vel, max_vel])
    angle = draw_from_range([0, 2*math.pi])

    x_dot = speed*math.cos(angle)
    y_dot = speed*math.sin(angle)

    return x_dot, y_dot


def draw_from_range(bounds):
    draw = bounds if type(bounds) != list else np.random.uniform(bounds[0], bounds[1])
    return draw
