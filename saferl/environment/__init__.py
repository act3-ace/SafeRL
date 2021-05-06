from saferl.environment import utils, models, tasks, callbacks

import inspect

callbacks_mems = inspect.getmembers(callbacks, inspect.isclass)
utils_mems = inspect.getmembers(utils, inspect.isclass)
mems = callbacks_mems + utils_mems

lookup = {v.__module__ + "." + k: v for k, v in mems if "saferl" in str(v)}
lookup = {**lookup, **models.lookup, **tasks.lookup}
