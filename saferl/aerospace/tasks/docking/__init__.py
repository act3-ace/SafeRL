import saferl.aerospace.tasks.docking.processors
import saferl.aerospace.tasks.docking.task

import inspect

processors_members = inspect.getmembers(processors, inspect.isclass)
task_members = inspect.getmembers(task, inspect.isclass)

mems = processors_members + task_members

lookup = {k: v for k, v in mems}

