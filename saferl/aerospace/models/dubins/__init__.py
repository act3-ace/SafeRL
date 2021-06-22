import inspect
from saferl.aerospace.models.dubins import platforms
from saferl.aerospace.models.dubins import rta

platform_mems = inspect.getmembers(platforms, inspect.isclass)
rta_mems = inspect.getmembers(rta, inspect.isclass)
mems = platform_mems + rta_mems

lookup = {v.__module__ + "." + k: v for k, v in mems if "saferl" in str(v)}
