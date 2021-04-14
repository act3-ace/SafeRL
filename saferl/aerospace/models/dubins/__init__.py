import inspect
import saferl.aerospace.models.dubins.platforms

lookup = {k: v for k, v in inspect.getmembers(platforms, inspect.isclass)}
