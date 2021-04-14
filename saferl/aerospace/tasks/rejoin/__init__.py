import saferl.aerospace.tasks.rejoin.processors
import saferl.aerospace.tasks.rejoin.task

import inspect

processors_members = inspect.getmembers(processors, inspect.isclass)
task_members = inspect.getmembers(task, inspect.isclass)

mems = processors_members + task_members

lookup = {k: v for k, v in mems}
