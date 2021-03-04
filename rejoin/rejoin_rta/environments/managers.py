class Manager:
    def __init__(self, config):
        self.config = config
        self.step_value = 0
        self.total_value = 0

        # Register and initialize processors
        self.processors = [p(config=config) for p in config["processors"]]

        # Initialize components
        self.components = {p.name: 0 for p in self.processors}

    def step(self, env_objs, timestep, status):
        self.step_value = 0
        for processor in self.processors:
            processor.step(env_objs=env_objs, timestep=timestep, status=status)
            self.step_value += processor.step_value
            self.components[processor.name] += processor.step_value
        self.total_value += self.step_value
        return self.step_value

    def _generate_info(self):
        info = {
            'step': self.step_value,
            'component_totals': self.components,
            'total': self.total_value,
        }

        return info

    def reset(self, env_objs):
        self.components = {p.name: 0 for p in self.processors}
        self.step_value = 0
        self.total_value = 0
        for p in self.processors:
            p.reset(env_objs)


class StatusManager(Manager):
    def __init__(self, config):
        super().__init__(config=config)
        self.measurement_manager = MeasurementManager(config=config)
        self.condition_manager = ConditionManager(config=config)
        self.status = {
            "measurements": self.measurement_manager.components,
            "conditions": self.condition_manager.components,
        }

    def step(self, env_objs, timestep, status):
        self.step_value = 0
        self.step_value += self.measurement_manager.step(env_objs=env_objs, timestep=timestep, status=status)
        self.step_value += self.condition_manager.step(env_objs=env_objs, timestep=timestep, status=status)
        return self.step_value


class MeasurementManager(Manager):
    def __init__(self, config):
        super().__init__(config=config)


class ConditionManager(Manager):
    def __init__(self, config):
        super().__init__(config=config)


class RewardManager(Manager):
    def __init__(self, config):
        super().__init__(config=config)
