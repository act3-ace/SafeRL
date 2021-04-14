import saferl.environment.tasks.env
import saferl.environment.tasks.manager
import saferl.environment.tasks.processor
import saferl.environment.tasks.utils

import inspect

env_mems = inspect.getmembers(env, inspect.isclass)
manager_mems = inspect.getmembers(manager, inspect.isclass)
processor_mems = inspect.getmembers(processor, inspect.isclass)
utils_mems = inspect.getmembers(utils, inspect.isclass)
mems = env_mems + manager_mems + utils_mems + processor_mems

lookup = {k: v for k, v in mems}
