import inspect
import saferl.aerospace.models.cwhspacecraft.platforms

mems = inspect.getmembers(platforms, inspect.isclass)

lookup = {__name__ + ".platforms." + k: v for k, v in mems if "saferl" in str(v)}
# lookup = {v.__module__ + "." + k: v for k, v in mems if "saferl" in str(v)}
