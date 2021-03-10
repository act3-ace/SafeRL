from abc import ABC, abstractmethod


# TODO: Implement observation manager


class Manager(ABC):
    def __init__(self, config):
        self.config = config

        # Register and initialize processors
        self.processors = [p(config=config) for p in config["processors"]]

    def reset(self, env_objs):
        for p in self.processors:
            p.reset(env_objs)

    def step(self, env_objs, timestep, status, old_status):
        for processor in self.processors:
            self._handle_processor(
                processor=processor,
                env_objs=env_objs,
                timestep=timestep,
                status=status,
                old_status=old_status
            )

    @abstractmethod
    def _generate_info(self) -> dict:
        """Create and return an info dict"""
        raise NotImplementedError

    @abstractmethod
    def _handle_processor(self, processor, env_objs, timestep, status, old_status):
        """Handle processor"""
        raise NotImplementedError


class ObservationManager(Manager):
    def __init__(self, config):
        super().__init__(config=config)

    def _generate_info(self) -> dict:
        info = {}
        return info

    def _handle_processor(self, processor, env_objs, timestep, status, old_status):
        processor.step(
            env_objs=env_objs,
            timestep=timestep,
            status=status,
            old_status=old_status
        )


class StatusManager(Manager):
    def __init__(self, config):
        super().__init__(config=config)
        self.status = {}

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.status = {}

    def _generate_info(self) -> dict:
        info = {
            'status': self.status
        }
        return info

    def _handle_processor(self, processor, env_objs, timestep, status, old_status):
        processor.step(
            env_objs=env_objs,
            timestep=timestep,
            status=status,
            old_status=old_status
        )
        self.status = status


class RewardManager(Manager):
    def __init__(self, config):
        super().__init__(config=config)
        self.step_value = 0
        self.total_value = 0

        # Initialize components
        self.components = {p.name: 0 for p in self.processors}

    def reset(self, env_objs):
        super().reset(env_objs=env_objs)
        self.components = {p.name: 0 for p in self.processors}

    def _generate_info(self):
        info = {
            'step': self.step_value,
            'component_totals': self.components,
            'total': self.total_value,
        }

        return info

    def step(self, env_objs, timestep, status, old_status):
        self.step_value = 0
        super().step(
            env_objs=env_objs,
            timestep=timestep,
            status=status,
            old_status=old_status
        )
        self.total_value += self.step_value

    def _handle_processor(self, processor, env_objs, timestep, status, old_status):
        processor.step(
            env_objs=env_objs,
            timestep=timestep,
            status=status,
            old_status=old_status
        )
        self.step_value += processor.step_value
