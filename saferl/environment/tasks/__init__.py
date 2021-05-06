from saferl.environment.tasks import env, manager, processor, utils

import inspect

env_mems = inspect.getmembers(env, inspect.isclass)
manager_mems = inspect.getmembers(manager, inspect.isclass)
processor_mems = inspect.getmembers(processor, inspect.isclass)
utils_mems = inspect.getmembers(utils, inspect.isclass)
mems = env_mems + manager_mems + utils_mems + processor_mems

lookup = {v.__module__ + "." + k: v for k, v in mems if "saferl" in str(v)}
