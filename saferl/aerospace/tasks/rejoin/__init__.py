from saferl.aerospace.tasks.rejoin import processors, task

import inspect

processors_members = inspect.getmembers(processors, inspect.isclass)
task_members = inspect.getmembers(task, inspect.isclass)

mems = processors_members + task_members

lookup = {v.__module__ + "." + k: v for k, v in mems if "saferl" in str(v)}
