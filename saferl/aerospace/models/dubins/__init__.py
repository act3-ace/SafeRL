import inspect
from saferl.aerospace.models.dubins import platforms

mems = inspect.getmembers(platforms, inspect.isclass)
lookup = {v.__module__ + "." + k: v for k, v in mems if "saferl" in str(v)}
