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

### Training
SafeRL is built around using config files to define environments. The included training script `scripts/train.py` can be used to construct an environment and train an agent directly from a yaml config file.

The following commands will train an rllib agent on one of our baseline environments
```shell
# Dubins aircraft rejoin baseline
python scripts/train.py --config configs/rejoin/rejoin_default.yaml

# Clohessy-Wiltshire spacecraft Docking baseline
python scripts/train.py --config configs/docking/docking_default.yaml
```

See ```python scripts/train.py --help``` for more training options.

### Evaluation
There are two options for evaluating the performance of an agent's policy: during training or after training.
To periodically evaluate policy during training, use the 'evaluation_during_training' boolean flag while running the training script:
```shell
python scripts/train.py --config configs/rejoin/rejoin_default.yaml --evaluation_during_training=True
```
For more control over evaluation rollouts during training, try setting some of the following arguments: evaluation_during_training, evaluation_interval, evaluation_num_episodes, evaluation_num_workers, evaluation_seed, and evaluation_exploration.

If you would like to evaluate a policy from a training run that has already completed, use the `scripts/eval.py` script.
Only the 'dir' flag is required, which defines the full path to where your experiment directory is located:
```shell
python scripts/eval.py --dir full/path/to/experiment_dir
```
You may wish to evaluate the policy of a specific saved checkpoint, in which case simply use the `ckpt_num` flag to pass the number of the checkpoint you wish to evaluate
```shell
python scripts/eval.py --dir full/path/to/experiment_dir --ckpt_num=200
```

A lot of the same options for evaluation during training are available for evaluation after training via command line arguments for `scripts/eval.py`.
See ```python scripts/eval.py --help``` for the full list.


## Documentation

General code documentation guidelines:
1. Use [SciPy/NumPy](https://numpydoc.readthedocs.io/en/latest/format.html) style docstrings for all front-facing classes and functions ([example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html)).
2. Use Python line comments to explain potentially obscure implementation details where they occur.
3. Use descriptive variable names.
4. Avoid using the same variable name for different purposes within the same scope.

Instructions on setting the NumPy docstring format as your default in PyCharm can be found [here](https://www.jetbrains.com/help/pycharm/settings-tools-python-integrated-tools.html).

## Team
Jamie Cunningham,
John McCarroll,
Nate Hamilton,
Kerianne Hobbs,
Umberto Ravaioli,
