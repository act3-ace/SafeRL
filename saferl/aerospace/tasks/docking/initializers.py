import math
import numpy as np

from saferl.environment.tasks.initializers import Initializer


class ConstrainedDeputyPolarInitializer(Initializer):
    def get_init_params(self):
        # Get spatial mode
        mode = self.init_config.get('mode', '2d')

        # Get reference
        ref = self.init_config["ref"]

        # Get radius
        radius = self.init_config["radius"]
        radius = radius if type(radius) != list else np.random.uniform(radius[0], radius[1])

        # Get angle
        angle = self.init_config["angle"]
        angle = angle if type(angle) != list else np.random.uniform(angle[0], angle[1])

        if mode == '3d':
            polar_angle = self.init_config.get('polar_angle', np.pi/2)
            polar_angle = draw_from_range(polar_angle)
        elif mode == '2d':
            polar_angle = np.pi/2
        else:
            raise ValueError("mode {} is invalid. Must be one of ('2d', '3d')".format(mode))

        x, y, z = get_relative_rect_from_polar(ref, radius, angle, polar_angle)

        x_dot, y_dot, z_dot = get_constrainted_velocity(ref, x, y, z, mode=mode)

        new_params = {
            "x": x,
            "y": y,
            "x_dot": x_dot,
            "y_dot": y_dot
        }

        if mode == '3d':
            new_params['z'] = z
            new_params['z_dot'] = z_dot

        return new_params


def get_relative_rect_from_polar(ref, radius, angle, polar_angle):
    ref_x, ref_y, ref_z = ref.x, ref.y, ref.z
    x = ref_x + radius * np.cos(angle) * np.sin(polar_angle)
    y = ref_y + radius * np.sin(angle) * np.sin(polar_angle)
    z = ref_z + radius * np.cos(polar_angle)
    return x, y, z


def get_constrainted_velocity(ref, x, y, z, mean_motion=0.001027, max_vel_constraint=10, mode='2d'):
    rel_x = x - ref.x
    rel_y = y - ref.y
    rel_z = z - ref.z

    # Velocity bound based on relative position
    # TODO: get rid of hardcoded values in constraint
    park_orbit_max_vel = 0.2 + 2 * mean_motion * math.sqrt(rel_x**2 + rel_y**2 + rel_z**2)
    park_orbit_min_vel = 0

    max_vel = min(park_orbit_max_vel, max_vel_constraint)
    min_vel = park_orbit_min_vel

    speed = draw_from_range([min_vel, max_vel])
    angle = draw_from_range([0, 2*math.pi])

    if mode == '2d':
        polar_angle = np.pi/2
    elif mode == '3d':
        polar_angle = draw_from_range([0, np.pi])
    else:
        raise ValueError("mode {} is invalid. Must be one of ('2d', '3d')".format(mode))

    x_dot = speed * math.cos(angle) * np.sin(polar_angle)
    y_dot = speed * math.sin(angle) * np.sin(polar_angle)
    z_dot = speed * math.cos(polar_angle)

    return x_dot, y_dot, z_dot


def draw_from_range(bounds):
    draw = bounds if type(bounds) != list else np.random.uniform(bounds[0], bounds[1])
    return draw
