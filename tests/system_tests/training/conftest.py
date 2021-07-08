"""
This module implements all fixtures common to testing the training functionality of our benchmarks.
"""

import pytest
import os
import jsonlines
import statistics
import shutil
import ray.rllib.agents.ppo as ppo
from ray import tune

from saferl.environment.callbacks import build_callbacks_caller, EpisodeOutcomeCallback, FailureCodeCallback,\
    RewardComponentsCallback, LoggingCallback, LogContents
from saferl import lookup
from saferl.environment.utils import YAMLParser


# define defaults
default_log_contents = (LogContents.INFO,)
default_gpus = 0
default_workers = 6
default_fake_gpus = False
default_seed = 100
default_output = "../../test_data/training_output"
# TODO: use tmp_dir and tmp_file fixtures


@pytest.fixture()
def success_threshold():
    default_success_threshold = 0.05
    return default_success_threshold


@pytest.fixture
def config(config_path):
    """
    This fixture parses the yaml file found at the specified config_path and returns the resulting config dict.
    If the file does not exist, a FileNotFound error is raised.

    Parameters
    ----------
    config_path : str
        The full path of the desired experiment's config file.

    Returns
    -------
    config : dict
        A map of experiment config settings.
    """

    if os.path.exists(config_path):
        # Setup default PPO config
        config = ppo.DEFAULT_CONFIG.copy()
        config["num_gpus"] = default_gpus
        config["num_workers"] = default_workers
        config['_fake_gpus'] = default_fake_gpus
        config['seed'] = default_seed
        config['callbacks'] = build_callbacks_caller([EpisodeOutcomeCallback(),
                                                      FailureCodeCallback(),
                                                      RewardComponentsCallback(),
                                                      LoggingCallback(num_logging_workers=default_workers,
                                                                  episode_log_interval=1,
                                                                  contents=default_log_contents)])

        # Setup custom config
        parser = YAMLParser(yaml_file=config_path, lookup=lookup)
        env, env_config = parser.parse_env()
        config['env'] = env
        config['env_config'] = env_config

        return config
    else:
        raise FileNotFoundError("Unable to locate: {}".format(config_path))


@pytest.fixture
def output_dir():
    """
    This fixture creates, returns, and tears down the output directory for test training logs, based on the
    default_output.

    Returns
    ----------
    output_path : str
        The full path of the desired experiment's config file.
    """

    output_path = os.path.join(os.getcwd(), default_output)
    os.makedirs(output_path, exist_ok=True)
    yield output_path
    shutil.rmtree(output_path)


@pytest.fixture
def training_output(config, stop_dict, output_dir, expr_name):
    """
    This fixture runs training, populating the output_path with training logs.

    Parameters
    ----------
    config : dict
        A map of experiment config settings.
    stop_dict : dict
        A dict specifying stop criteria for the experiment.
    output_dir : Fixture
        The full path of the desired experiment's config file.
    expr_name : str
        The name of the experiment to run.
    """

    #TODO: .run returns results* (use results var and remove logging and cleanup)
    results = tune.run(ppo.PPOTrainer, config=config, stop=stop_dict, local_dir=output_dir,
                checkpoint_at_end=True, name=expr_name)
    return results


@pytest.fixture
def success_rate(training_output, output_dir, expr_name):
    """
    This fixture parses a training log and returns the success ratio.

    Parameters
    ----------
    training_output : Fixture
        A fixture that runs the experiment in order to produce logs.
    output_dir : str
        The full path of the desired experiment's config file.
    expr_name : str
        The name of the experiment to run.

    Returns
    -------
    success_rate : float
        The ratio of the successes to failures.
    """

    # parse log file
    successes = list()
    episodes = set()
    log_file = os.path.join(output_dir, expr_name, "training_logs/worker_1.log")
    with jsonlines.open(log_file, 'r') as log:
        prev_ID = None
        prev_success = None

        # iterate through json objects in log
        for state in log:
            episode_success = state["info"]["success"]
            episode_ID = state["episode_ID"]

            if episode_ID not in episodes:
                # start of new episode
                episodes.add(episode_ID)
                # store previous episode's success data (1 = success, 0 = failure)
                if prev_ID:
                    successes.append(int(prev_success))

            prev_ID = episode_ID
            prev_success = episode_success

        # add last episode's data
        successes.append(int(episode_success))
        success_rate = statistics.mean(successes)

        return success_rate
