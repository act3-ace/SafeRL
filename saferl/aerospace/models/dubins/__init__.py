import inspect
import saferl.aerospace.models.dubins.platforms

mems = inspect.getmembers(platforms, inspect.isclass)
lookup = {v.__module__ + "." + k: v for k, v in mems if "saferl" in str(v)}
