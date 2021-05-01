# Intro
The SafeRL library provides the components and tools to build modular, OpenAI Gym compatible Reinforement Learning environments with Run Time Assurance (RTA). SafeRL is designed to work best with ray/rllib.

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
SafeRL is built around using config files to define environments. The included training script `scripts/train.py` can be used to construct an environment and train an agent directly from a yaml config file.

The following commands will train an rllib agent on one of our baseline environments
```shell
# Dubins aircraft rejoin baseline
python scripts/train.py configs/rejoin/rejoin_default.yaml

# Clohessy-Wiltshire spacecraft Docking baseline
python scripts/train.py configs/docking/docking_default.yaml
```

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
