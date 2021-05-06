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
  
  # Define a list of environment platforms
  env_objs:
    # First environment platform
  - name: deputy  # Platform name
    class: saferl.aerospace.models.cwhspacecraft.platforms.CWHSpacecraft2d  # Platform class
    config:  # Platform dependent configuration
      controller:
        actuators:
        - name: thrust_x
          space: discrete
          points: 11
          bounds: [-10, 10]
        - name: thrust_y
          space: discrete
          points: 11
          bounds: [-10, 10]
      init:  # Platform initialization parameters
        x: [-2000, 2000]
        x_dot: 0
        y: [-2000, 2000]
        y_dot: 0
  - name: chief  # Platform name
    class: saferl.aerospace.models.cwhspacecraft.platforms.CWHSpacecraft2d  # Platform class
    config:  # Platform dependent configuration
      init:  # Platform initialization parameters
        x: 0
        x_dot: 0
        y: 0
        y_dot: 0
  - name: docking_region  # Platform name
    class: saferl.environment.models.geometry.RelativeCircle  # Platform class
    config:  # Platform dependent configuration
      ref: chief
      x_offset: 0
      y_offset: 0
      radius: 20
  observation:  # List of observation processors
  - name: observation_processor  # Observation processor name
    class: saferl.aerospace.tasks.docking.processors.DockingObservationProcessor  # Processor class
    config:  # Processor-dependent configuration
      deputy: deputy
      mode: 2d
  reward:  # List of reward processors
  - name: time_reward  # Processor name
    class: saferl.aerospace.tasks.docking.processors.TimeRewardProcessor  # Processor class
    config:  # Processor-dependent configuration
      reward: -0.01
  - name: dist_change_reward  # Processor name
    class: saferl.aerospace.tasks.docking.processors.DistanceChangeRewardProcessor  # Processor class
    config:  # Processor-dependent configuration
      deputy: deputy
      docking_region: docking_region
      reward: -1.0e-03
  - name: failure_reward  # Processor name
    class: saferl.aerospace.tasks.docking.processors.FailureRewardProcessor  # Processor class
    config:  # Processor-dependent configuration
      failure_status: failure
      reward:
        crash: -1
        distance: -1
        timeout: -1
  - name: success_reward  # Processor name
    class: saferl.aerospace.tasks.docking.processors.SuccessRewardProcessor  # Processor class
    config:  # Processor-dependent configuration
      reward: 1
      success_status: success
  status:  # List of observation processors
  - name: docking_status  # Processor name
    class: saferl.aerospace.tasks.docking.processors.DockingStatusProcessor  # Processor class
    config:  # Processor-dependent configuration
      deputy: deputy
      docking_region: docking_region
  - name: docking_distance  # Processor name
    class: saferl.aerospace.tasks.docking.processors.DockingDistanceStatusProcessor  # Processor class
    config:  # Processor-dependent configuration
      deputy: deputy
      docking_region: docking_region
  - name: failure  # Processor name
    class: saferl.aerospace.tasks.docking.processors.FailureStatusProcessor  # Processor class
    config:  # Processor-dependent configuration
      docking_distance: docking_distance
      max_goal_distance: 40000
      timeout: 1000
  - name: success  # Processor name
    class: saferl.aerospace.tasks.docking.processors.SuccessStatusProcessor  # Processor class
    config:  # Processor-dependent configuration
      docking_status: docking_status
  verbose: false  # Training verbosity
```

## Env

```yaml
env: saferl.aerospace.tasks.docking.task.DockingEnv
```

The ```env``` key defines the Gym environment used during training.
Every ```env``` object expects an ```env_config``` configuration.

## Env Config
