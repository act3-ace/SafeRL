import saferl.environment.utils
import saferl.environment.models
import saferl.environment.tasks
import saferl.environment.callbacks

import inspect

callbacks_mems = inspect.getmembers(callbacks, inspect.isclass)
utils_mems = inspect.getmembers(utils, inspect.isclass)
mems = callbacks_mems + utils_mems

lookup = {k: v for k, v in mems}
lookup = {**lookup, **models.lookup, **tasks.lookup}
