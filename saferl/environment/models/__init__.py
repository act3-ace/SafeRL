import saferl.environment.models.platforms
import saferl.environment.models.geometry

import inspect

platforms_mems = inspect.getmembers(platforms, inspect.isclass)
geometry_mems = inspect.getmembers(geometry, inspect.isclass)
mems = platforms_mems + geometry_mems

lookup = {v.__module__ + "." + k: v for k, v in mems if "saferl" in str(v)}
