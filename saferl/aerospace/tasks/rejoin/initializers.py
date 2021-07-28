import numpy as np

from saferl.environment.tasks.initializers import Initializer


class WingmanPolarInitializer(Initializer):
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

        new_params = {
            "x": x,
            "y": y
        }
        for k, v in self.init_config.items():
            new_params[k] = v if type(v) != list else np.random.uniform(v[0], v[1])
        return new_params


def get_relative_rect_from_polar(reference, radius, angle):
    ref_x, ref_y = reference.x, reference.y
    x = ref_x + radius * np.cos(angle)
    y = ref_y + radius * np.sin(angle)
    return x, y
