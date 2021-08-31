# Intro
The SafeRL library provides the components and tools to build modular, OpenAI Gym compatible Reinforcement Learning environments with Run Time Assurance (RTA). SafeRL is designed to work best with ray/rllib.

## Installation
Inside of the repo's root directory, simply install using the `setup.py` with:
```shell
pip install .
```

For a local development version, please install using the `-e, --editable` option:
```shell
pip install -e .
```

## Usage

### Config Files
The SafeRL library works via [config files](docs/config_docs.md). These config files define the task environment, the simulation objects within the environment, and the rewards/observations passed to the agent. This allows environments and experiments to be shared, recreated, and reconfigured. New RL experiments may be implemented and executed simply through manipulation of config files. More custom behavior (such as a custom reward) can be implemented via small pieces of python code and integrated with an existing environment via the config file.

Config files may also contain learning parameters for the RL algorithm. In the case of `train.py` learning parameters for Ray RLlib are automatically passed, however should you choose to use an alternative RL framework, simply extract the parameters from the dictionary returned by the config parser.

### Training
The included training script `scripts/train.py` can be used to construct an environment and train an agent directly from a yaml config file. This training script uses Ray RLlib and Ray Tune, however our environments can be integrated into any OpenAI Gym compatible RL framework/implementation.

The following commands will train an rllib agent on one of our baseline environments
```shell
# Dubins aircraft rejoin baseline
python scripts/train.py --config configs/rejoin/rejoin_default.yaml

# Clohessy-Wiltshire spacecraft Docking baseline
python scripts/train.py --config configs/docking/docking_default.yaml --stop_iteration 2000 --complete_episodes
```

See ```python scripts/train.py --help``` for more training options.

### Evaluation
There are two options for evaluating the performance of an agent's policy: during training or after training.
To periodically evaluate policy during training, use the 'evaluation_during_training' boolean flag while running the 
training script:
```shell
python scripts/train.py --config configs/rejoin/rejoin_default.yaml --eval
```
For more control over evaluation rollouts during training, try setting some of the following arguments: 
evaluation_during_training, evaluation_interval, evaluation_num_episodes, evaluation_num_workers, evaluation_seed, and 
evaluation_exploration.

If you would like to evaluate a policy from a training run that has already completed, use the `scripts/eval.py` script.
Only the 'dir' flag is required, which defines the full path to where your experiment directory is located:
```shell
python scripts/eval.py --dir full/path/to/experiment_dir
```
You may wish to evaluate the policy of a specific saved checkpoint, in which case simply use the `ckpt_num` flag to pass 
the number of the checkpoint you wish to evaluate
```shell
python scripts/eval.py --dir full/path/to/experiment_dir --ckpt_num=200
```

A lot of the same options for evaluation during training are available for evaluation after training via command line 
arguments for `scripts/eval.py`. The evaluation script even offers some additional functionality, including rendering
animations for evaluation episodes. See ```python scripts/eval.py --help``` for the full list of options or read our 
documentation on evaluation and animation [here](docs/animation/evaluation_and_animation.md) for more details.




## Environments

### Rejoin
Aircraft formation flight rejoin where a wingman aircraft controlled by the agent must join a formation relative to a lead aircraft. The formation is defined by a rejoin region relative to the lead's position and orientation which the wingman must enter and remain within. Comes in the following flavors:

-  **Rejoin 2D**  
Throttle and heading control.  
Config file: `configs/docking/rejoin_default.yaml`  

-  **Rejoin 3D**  
Throttle, heading, flight angle, roll control.  
Config files: 
    - `configs/docking/rejoin_3d_default_continuous.yaml`  
    - `configs/docking/rejoin_3d_default_discrete.yaml`

### Docking
Spacecraft docking scenario where an agent controlled deputy spacecraft must dock with a stationary chief spacecraft while both orbit a central body. This is accomplished by approaching the chief to within a predefined docking distance while maintaining a safe relative velocity within that distance. The motion of the deputy spacecraft is governed by the Clohessy-Wiltshire linearlized dynamics model. Comes in the following flavors:

-  **Docking 2D**  
Static 1N thrusters in $`\pm x`$ and  $`\pm y`$.  
Config file: `configs/docking/docking_default.yaml`  

-  **Docking 3D**
Static 1N thrusters in $`\pm x, \pm y, \pm z`$. 
Does not currently train to successful completion.  
Config file: `configs/docking/docking_3d_default.yaml`  

## Documentation

General code documentation guidelines:
1. Use [SciPy/NumPy](https://numpydoc.readthedocs.io/en/latest/format.html) style docstrings for all front-facing classes and functions ([example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)).
2. Use Python line comments to explain potentially obscure implementation details where they occur.
3. Use descriptive variable names.
4. Avoid using the same variable name for different purposes within the same scope.

Instructions on setting the NumPy docstring format as your default in PyCharm can be found [here](https://www.jetbrains.com/help/pycharm/settings-tools-python-integrated-tools.html).

## Public Release
Approved for public release: distribution unlimited. Case Numbers: AFRL-2021-0064 and AFRL-2021-0065

## Team
Jamie Cunningham,
John McCarroll,
Kyle Dunlap,
Kerianne Hobbs,
Umberto Ravaioli,
Vardaan Gangal
