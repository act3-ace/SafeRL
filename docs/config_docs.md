# Configuration File Structure

## Complete File

Below is an example of a complete configuration file. The following
sections will explain each part of the configuration file in more 
detail.

```yaml
# Specify task environment
env: saferl.aerospace.tasks.docking.task.DockingEnv

# Task environment configuration
env_config:
  
  # Define the name of the agent platform
  agent: deputy
  
  # Define simulation step size
  step_size: 1
  
  # Define a list of environment platforms
  env_objs:
  # First environment platform
  - name: chief # Platform name
    class: saferl.aerospace.models.cwhspacecraft.platforms.CWHSpacecraft2d # Platform class
    config: # Platform dependent configuration
      init: # Platform initialization parameters
        initializer: saferl.environment.tasks.initializers.RandBoundsInitializer
        x: 0
        x_dot: 0
        y: 0
        y_dot: 0
  - name: deputy # Platform name
    class: saferl.aerospace.models.cwhspacecraft.platforms.CWHSpacecraft2d # Platform class
    config: # Platform dependent configuration
      controller:
        class: saferl.environment.models.platforms.AgentController
        actuators:
        - name: thrust_x
          space: discrete
          points: 11
          bounds: [-1, 1]
        - name: thrust_y
          space: discrete
          points: 11
          bounds: [-1, 1]
      init: # Platform initialization parameters
        initializer: saferl.aerospace.tasks.docking.initializers.ConstrainedDeputyPolarInitializer
        ref: chief
        radius: [100, 150]
        angle: [0, 6.283185307179586]
  - name: docking_region # Platform name
    class: saferl.environment.models.geometry.RelativeCircle # Platform class
    config: # Platform dependent configuration
      ref: chief
      x_offset: 0
      y_offset: 0
      radius: 0.5
      init: # Platform initialization parameters
        initializer: saferl.environment.tasks.initializers.RandBoundsInitializer
  status: # List of status processors
    - name: in_docking # Processor name
      class: saferl.aerospace.tasks.docking.processors.InDockingStatusProcessor # Processor class
      config: # Processor-dependent configuration
        deputy: deputy
        docking_region: docking_region
    - name: docking_distance # Processor name
      class: saferl.aerospace.tasks.docking.processors.DockingDistanceStatusProcessor # Processor class
      config: # Processor-dependent configuration
        deputy: deputy
        docking_region: docking_region
    - name: delta_v # Processor name
      class: saferl.aerospace.tasks.docking.processors.DockingThrustDeltaVStatusProcessor # Processor class
      config: # Processor-dependent configuration
        target: deputy
    - name: custom_metrics.delta_v_total # Processor name
      class: saferl.aerospace.tasks.docking.processors.AccumulatorStatusProcessor # Processor class
      config: # Processor-dependent configuration
        status: delta_v
    - name: max_vel_limit # Processor name
      class: saferl.aerospace.tasks.docking.processors.DockingVelocityLimit # Processor class
      config: # Processor-dependent configuration
        target: deputy
        dist_status: docking_distance
        vel_threshold: 0.2
        threshold_dist: 0.5
        slope: 2
    - name: max_vel_violation # Processor name
      class: saferl.aerospace.tasks.docking.processors.DockingVelocityLimitViolation # Processor class
      config: # Processor-dependent configuration
        target: deputy
        ref: chief
        vel_limit_status: max_vel_limit
    - name: max_vel_constraint # Processor name
      class: saferl.aerospace.tasks.docking.processors.RelativeVelocityConstraint # Processor class
      config: # Processor-dependent configuration
        target: deputy
        ref: chief
        vel_limit_status: max_vel_limit
    - name: failure # Processor name
      class: saferl.aerospace.tasks.docking.processors.FailureStatusProcessor # Processor class
      config: # Processor-dependent configuration
        docking_distance: docking_distance
        max_goal_distance: 40000
        timeout: 2000
        in_docking_status: in_docking
        max_vel_constraint_status: max_vel_constraint
    - name: success # Processor name
      class: saferl.aerospace.tasks.docking.processors.SuccessStatusProcessor # Processor class
      config: # Processor-dependent configuration
        in_docking_status: in_docking
        max_vel_constraint_status: max_vel_constraint
  observation: # List of observation processors
    - name: observation_processor # Processor name
      class: saferl.aerospace.tasks.docking.processors.DockingObservationProcessor # Processor class
      config: # Processor-dependent configuration
        deputy: deputy
        mode: 2d
  reward: # List of reward processors
    - name: time_reward # Processor name
      class: saferl.aerospace.tasks.docking.processors.TimeRewardProcessor # Processor class
      config: # Processor-dependent configuration
        reward: -0.001
    - name: dist_change_reward # Processor name
      class: saferl.aerospace.tasks.docking.processors.DistanceChangeRewardProcessor # Processor class
      config: # Processor-dependent configuration
        deputy: deputy
        docking_region: docking_region
        reward: -1.0e-02
    - name: delta_v # Processor name
      class: saferl.environment.tasks.processor.reward.ProportionalRewardProcessor # Processor class
      config: # Processor-dependent configuration
        scale: 0
        bias: 0
        proportion_status: delta_v
    - name: max_vel_constraint # Processor name
      class: saferl.environment.tasks.processor.reward.ProportionalRewardProcessor # Processor class
      config: # Processor-dependent configuration
        scale: -0.001
        bias: -0.01
        proportion_status: max_vel_violation
        cond_status: max_vel_constraint
        cond_status_invert: True
    - name: failure_reward # Processor name
      class: saferl.aerospace.tasks.docking.processors.FailureRewardProcessor # Processor class
      config: # Processor-dependent configuration
        failure_status: failure
        reward:
          crash: -1
          distance: -1
          timeout: -5
    - name: success_reward # Processor name
      class: saferl.aerospace.tasks.docking.processors.SuccessRewardProcessor # Processor class
      config: # Processor-dependent configuration
        reward: 5
        success_status: success
  verbose: false # Training verbosity

```

## Env

```yaml
env: saferl.aerospace.tasks.docking.task.DockingEnv
```

The ```env``` key defines the Gym environment used during training.
Every ```env``` object expects an ```env_config``` configuration.
Specify the Gym environment you wish to use with the import path
of the environment class.

## Env Config
```yaml
env_config:
```
The env_config key points to the custom environment configuration
for the environment specified by ```env```. All environments which
inherit from the SafeRL ```BaseEnv``` class expect the follow keys
in the custom ```env_config```:

- ```agent: ``` a string listing the name of the environment object
which will take actions within the environment.
  
- `step_size: ` a float denoting the step size of the simulation.
  
- ```env_objs:``` an ordered list of ```EnvObj``` config entries. This list contains
every ```EnvObj``` present in the environment and their relative
configurations.
  
- ```observation:``` an ordered list of `ObservationProcessor` config entries.
This list contains every `ObservationProcessor` present in the environment and their relative
configurations.
  
- ```reward:``` an ordered list of `RewardProcessor` config entries.
This list contains every `RewardProcessor` present in the environment and their relative
configurations.
  
- ```status:``` an ordered list of `StatusProcessor` config entries.
This list contains every `StatusProcessor` present in the environment and their relative
configurations.
  
- `verbose:` A boolean flag which sets the verbosity of environment
setup and training.
  
Each of these entries in the `env_config` configuration will be
explained in further detail below.

### EnvObj Entries

```yaml
name: deputy # Platform name
class: saferl.aerospace.models.cwhspacecraft.platforms.CWHSpacecraft2d # Platform class
config: # Platform dependent configuration
  controller:
    class: saferl.environment.models.platforms.AgentController
    actuators:
    - name: thrust_x
      space: discrete
      points: 11
      bounds: [-1, 1]
    - name: thrust_y
      space: discrete
      points: 11
      bounds: [-1, 1]
  init: # Platform initialization parameters
    initializer: saferl.aerospace.tasks.docking.initializers.ConstrainedDeputyPolarInitializer
    ref: chief
    radius: [100, 150]
    angle: [0, 6.283185307179586]
```

EnvObj entries in a custom YAML configuration file consist of three
primary keys:

- `name`
- `class`
- `config`

#### Name

The name key points to a string value denoting the name of the
environment object in the simulation. The environment object can
be easily referenced by its name throughout the rest of the
configuration file.

#### Class

The class key points to a string value containing the python
import path of the class definition that will be used to instantiate
the environment object in the simulation. This class must inherit
from the `saferl.environment.models.platforms.BaseEnvObj` class.

#### Config

The config key points to a (potentially empty) dictionary of key-value
pairs. The contents of this configuration dictionary include an `init`
configuration and a configuration which is dependent
on the expected configuration parameters of the environment object
`class`. For example, the `saferl.aerospace.models.cwhspacecraft.platforms.CWHSpacecraft2d`
class can take a `controller` key-value configuration which
defines the controller class and actuator parameters for the platform.

The `config` dictionary also contains an `init` configuration. This
configuration specifies the `initializer` class used to initialize
the platform within the environment, as well as whatever custom
parameters the `initializer` class expects. The `initializer` class
is expected to inherit from `saferl.environment.tasks.initializers.Initializer`.


### Observation Processor Entries

```yaml
name: observation_processor # Processor name
class: saferl.aerospace.tasks.docking.processors.DockingObservationProcessor # Processor class
config: # Processor-dependent configuration
    deputy: deputy
    mode: 2d
```

Observation processor entries in a custom YAML configuration file consist of three
primary keys:

- `name`
- `class`
- `config`

#### Name

The name key points to a string value denoting the name of the
observation processor in the simulation. The processor can
be easily referenced by its name throughout the rest of the
configuration file.

#### Class

The class key points to a string value containing the python
import path of the class definition that will be used to instantiate
the observation processor in the simulation. This class must inherit
from the `saferl.environment.tasks.processor.processors.ObservationProcessor` class.

#### Config

The config key points to a (potentially empty) dictionary of key-value
pairs. The contents of this configuration dictionary include a configuration which is dependent
on the expected configuration parameters of the observation processor
`class`. For example, the ` saferl.aerospace.tasks.docking.processors.DockingObservationProcessor`
class can take a `deputy` key-value configuration, which registers
the name of the deputy environment object in the simulation, and a
`mode` key-value configuration which defines whether observations
are 2D or 3D.


### Reward Processor Entries

```yaml
name: time_reward # Processor name
class: saferl.aerospace.tasks.docking.processors.TimeRewardProcessor # Processor class
config: # Processor-dependent configuration
    reward: -0.001
```

Reward processor entries in a custom YAML configuration file consist of three
primary keys:

- `name`
- `class`
- `config`

#### Name

The name key points to a string value denoting the name of the
reward processor in the simulation. The processor can
be easily referenced by its name throughout the rest of the
configuration file.

#### Class

The class key points to a string value containing the python
import path of the class definition used to instantiate
the reward processor in the simulation. This class must inherit
from the `saferl.environment.tasks.processor.processors.RewardProcessor` class.

#### Config

The config key points to a (potentially empty) dictionary of key-value
pairs. The contents of this configuration dictionary include a configuration which is dependent
on the expected configuration parameters of the observation processor
`class`. For example, the ` saferl.aerospace.tasks.docking.processors.TimeRewardProcessor`
class can take a `reward` key-value configuration, which defines
the reward value associated with time taken to complete the task.


### Status Processor Entries

```yaml
name: in_docking # Processor name
class: saferl.aerospace.tasks.docking.processors.InDockingStatusProcessor # Processor class
config: # Processor-dependent configuration
    deputy: deputy
    docking_region: docking_region
```

Status processor entries in a custom YAML configuration file consist of three
primary keys:

- `name`
- `class`
- `config`

#### Name

The name key points to a string value denoting the name of the
status processor in the simulation. The processor can
be easily referenced by its name throughout the rest of the
configuration file.

#### Class

The class key points to a string value containing the python
import path of the class definition used to instantiate
the status processor in the simulation. This class must inherit
from the `saferl.environment.tasks.processor.processors.StatusProcessor` class.

#### Config

The config key points to a (potentially empty) dictionary of key-value
pairs. The contents of this configuration dictionary include a configuration which is dependent
on the expected configuration parameters of the observation processor
`class`. For example, the ` saferl.aerospace.tasks.docking.processors.InDockingStatusProcessor`
class can take a `deputy` key-value configuration, which registers
the name of the deputy environment object in the simulation, and a
`docking_region` key-value configuration which registers the name
of the docking region environment object in the simulation.
