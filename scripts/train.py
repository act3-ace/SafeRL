import argparse
import os
import yaml

from datetime import datetime

import ray
from ray import tune
from ray.tune.logger import TBXLoggerCallback

import ray.rllib.agents.ppo as ppo

from saferl.environment.utils import YAMLParser, build_lookup, dict_merge
from saferl.environment.callbacks import build_callbacks_caller, EpisodeOutcomeCallback, FailureCodeCallback, \
                                        RewardComponentsCallback, LoggingCallback, LogContents, \
                                        StatusCustomMetricsCallback, ConstraintViolationMetricsCallback


# Training defaults
CONFIG = '../configs/docking/docking_default.yaml'
OUTPUT_DIR = './output'
NUM_LOGGING_WORKERS = 1
LOGGING_INTERVAL = 10  # log every 10th episode
CONTENTS = (LogContents.VERBOSE,)
CUDA_VISIBLE_DEVICES = "-1"
NUM_GPUS = 0
NUM_WORKERS = 6
NUM_ENVS_PER_WORKER = 1
FAKE_GPUS = True
SEED = 0
STOP_ITERATION = 200
CHECKPOINT_FREQUENCY = 25
EVALUATION_INTERVAL = 50
EVALUATION_NUM_EPISODES = 10
EVALUATION_NUM_WORKERS = 1
EVALUATION_SEED = 0
DEBUG = False
COMPLETE = False
ROLLOUT_FRAGMENT_LENGTH = 200


def get_args():
    parser = argparse.ArgumentParser()

    # Add parser arguments
    parser.add_argument('--config', type=str, default=CONFIG, help="path to configuration file")
    parser.add_argument('--debug', default=DEBUG, action="store_true", help="set debug state to True")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help="path to output directory")
    parser.add_argument('--logging_workers', type=int, default=NUM_LOGGING_WORKERS,
                        help="number of workers for logging")
    parser.add_argument('--log_interval', type=int, default=LOGGING_INTERVAL, help="number of episodes between logging")
    parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQUENCY, help="tune checkpoint frequency")
    parser.add_argument('--cuda_visible', type=str, default=CUDA_VISIBLE_DEVICES, help="list of cuda visible devices")
    parser.add_argument('--gpus', type=int, default=None, help="number of gpus used for training")
    parser.add_argument('--workers', type=int, default=None, help="number of cpu workers used for training")
    parser.add_argument(
        '--envs_per_worker',
        type=int,
        default=None,
        help="number of environments per cpu worker used for training"
    )
    parser.add_argument('--fake_gpus', action="store_true", help="use simulated gpus")
    parser.add_argument('--seed', type=int, default=None, help="set random seed")
    parser.add_argument('--stop_iteration', type=int, default=STOP_ITERATION, help="number of iterations to run")
    parser.add_argument(
        '--restore',
        type=str,
        default=None,
        help="""restore agent from checkpoint and resume training with the passed config file.
                Must be path to tune checkpoint file within the checkpoint directory.
                E.g. --restore <path to experiment>/checkpoint_xx/checkpoint-xx"""
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help="""name of experiment to restore. Note that --output_dir must contain the resumed experiment directory.
                If no saved checkpoint is present, training will resume from first iteration.
                Make sure --checkpoint_freq is set to a reasonable value"""
    )

    parser.add_argument('--complete_episodes', action="store_true",
                        help="True if using complete episodes during training desired, "
                             "False if using truncated episodes")
    parser.add_argument('--rollout_fragment_length', type=int, default=None,
                        help="size of batches collected by each worker if truncated episodes")

    parser.add_argument('--eval', action="store_true",
                        help="True if intermittent evaluation of agent policy during training desired, False if not")
    parser.add_argument('--evaluation_interval', type=int, default=None,
                        help="number of episodes to run in between policy evaluations")
    parser.add_argument('--evaluation_num_episodes', type=int, default=None,
                        help="number of evaluation episodes to run")
    parser.add_argument('--evaluation_num_workers', type=int, default=None,
                        help="number of workers used to run evaluation episodes")
    parser.add_argument('--evaluation_seed', type=int, default=None,
                        help="set random seed for evaluation episodes")
    parser.add_argument('--evaluation_exploration', action="store_true",
                        help="set exploration behavior for evaluation episodes")

    parser.add_argument('--hpo_config', type=str, default=None)

    args = parser.parse_args()

    return args


def experiment_setup(args):
    # Set visible gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible

    # Create output directory and save filepaths
    expr_name = datetime.now().strftime("expr_%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, expr_name)
    os.makedirs(output_dir, exist_ok=True)
    args_yaml_filepath = os.path.join(output_dir, 'script_args.yaml')
    ray_config_yaml_filepath = os.path.join(output_dir, 'ray_config.yaml')

    # Initialize Ray
    ray.init(num_gpus=args.gpus)

    # Setup default PPO config
    default_config = ppo.DEFAULT_CONFIG.copy()

    # Setup custom config
    parser = YAMLParser(yaml_file=args.config, lookup=build_lookup())
    config = parser.parse_env()

    config_fill_with_arg(config, 'num_gpus', args.gpus, NUM_GPUS)
    config_fill_with_arg(config, 'num_workers', args.workers, NUM_WORKERS)
    config_fill_with_arg(config, 'seed', args.seed, SEED)
    config_fill_with_arg(config, 'num_envs_per_worker', args.envs_per_worker, NUM_ENVS_PER_WORKER)
    config_fill_with_arg(config, 'rollout_fragment_length', args.rollout_fragment_length, ROLLOUT_FRAGMENT_LENGTH)

    config['_fake_gpus'] = args.fake_gpus
    if args.complete_episodes:
        config['batch_mode'] = "complete_episodes"

    failure_codes = get_failure_codes(config)

    config['callbacks'] = build_callbacks_caller([EpisodeOutcomeCallback(),
                                                  FailureCodeCallback(failure_codes=failure_codes),
                                                  RewardComponentsCallback(),
                                                  LoggingCallback(num_logging_workers=args.logging_workers,
                                                                  episode_log_interval=args.log_interval,
                                                                  contents=CONTENTS),
                                                  StatusCustomMetricsCallback(),
                                                  ConstraintViolationMetricsCallback()])

    if args.eval:
        # set evaluation parameters
        config_fill_with_arg(config, 'evaluation_interval', args.evaluation_interval, EVALUATION_INTERVAL)
        config_fill_with_arg(config, 'evaluation_num_episodes', args.evaluation_num_episodes, EVALUATION_NUM_EPISODES)
        config_fill_with_arg(config, 'evaluation_num_workers', args.evaluation_num_workers, EVALUATION_NUM_WORKERS)

        if "evaluation_config" not in config:
            config["evaluation_config"] = {}
        config_fill_with_arg(config["evaluation_config"], 'seed', args.evaluation_seed, EVALUATION_SEED)

        if args.evaluation_exploration:
            config["evaluation_config"]['explore'] = True

        config["evaluation_config"]['callbacks'] = \
            build_callbacks_caller([EpisodeOutcomeCallback(),
                                    FailureCodeCallback(),
                                    RewardComponentsCallback(),
                                    LoggingCallback(num_logging_workers=args.evaluation_num_workers,
                                                    contents=CONTENTS)])

    # Merge custom and default configs
    config = dict_merge(default_config, config, recursive=True)

    # Save experiment params
    with open(args_yaml_filepath, 'w') as args_yaml_file:
        arg_dict = vars(args)
        yaml.dump(arg_dict, args_yaml_file)
    with open(ray_config_yaml_filepath, 'w') as ray_config_yaml_file:
        yaml.dump(config, ray_config_yaml_file)

    return expr_name, config


def config_fill_with_arg(config, key, arg, arg_default):
    if arg is not None:
        config[key] = arg
    elif key not in config:
        if arg is not None:
            config[key] = arg
        else:
            config[key] = arg_default


def get_failure_codes(config):
    try:
        reward_config_list = config['env_config']['reward']

        failure_codes = None
        for reward_config in reward_config_list:
            if reward_config['name'] == "failure_reward":
                failure_codes = reward_config['config']['reward'].keys()
    except KeyError:
        failure_codes = None

    if failure_codes is None:
        print("Failed to find failure_reward processor, defaulting to default failure code logging.")

    return failure_codes


def build_zoopt_search(search_alg_config, hpo_config, config):
    from ray.tune.suggest.zoopt import ZOOptSearch
    from zoopt import ValueType

    zoopt_valuetype_map = {
        'continuous': ValueType.CONTINUOUS,
        'discrete': ValueType.DISCRETE,
        'grid': ValueType.GRID,
    }

    zoopt_search_config = {
        'budget': hpo_config['num_samples'],  # must match `num_samples` in `tune.run()`.
        'parallel_num': config['num_workers'],  # how many workers to parallel
    }

    if 'dim_dict' in search_alg_config:
        dim_dict = search_alg_config.get('dim_dict', {})
        for _, dim_args in dim_dict.items():
            try:
                dim_args[0] = zoopt_valuetype_map[dim_args[0]]
            except KeyError as e:
                raise Exception(f"search space {dim_args[0]} is not available. \
                    Must be one of ['continuous', 'discrete', 'grid']") from e
            search_alg_config['dim_dict'] = dim_dict

    search_alg = ZOOptSearch(
        algo="Asracos",  # only support Asracos currently
        **{**zoopt_search_config, **search_alg_config},
    )

    return search_alg


def build_hpo_config(config, args):
    assert not (args.resume and args.hpo_config), "hyperparameter optimization currently not supported with resume"

    if args.hpo_config:
        with open(args.hpo_config, 'r') as f_hpo_config:
            hpo_config = yaml.safe_load(f_hpo_config)
    else:
        hpo_config = {}

    # construct search algorithm object from hpo config
    if 'search_alg' in hpo_config:
        search_alg_config = hpo_config['search_alg']
        search_alg = search_alg_config.pop('type', None)

        if search_alg == 'zoopt':
            search_alg = build_zoopt_search(search_alg_config, hpo_config, config)
        elif search_alg is not None:
            raise ValueError(f"search algorithm {search_alg} is not currently supported")

        hpo_config['search_alg'] = search_alg

    # construct scheduler object from hpo config
    if 'scheduler' in hpo_config:
        scheduler = hpo_config['scheduler'].pop('type', None)

        if scheduler == 'asha':
            from ray.tune.schedulers import AsyncHyperBandScheduler
            asha_config = {
                'max_t': args.stop_iteration,
            }

            scheduler = AsyncHyperBandScheduler(**{**asha_config, **hpo_config['scheduler']})
        elif scheduler is not None:
            raise ValueError(f"scheduler {scheduler} is not currently supported")

        hpo_config['scheduler'] = scheduler

    return hpo_config


def main(args):

    # Initialize stop dict
    stop_dict = {
        'training_iteration': args.stop_iteration,
    }

    # Run training
    if not args.debug:
        if args.resume:
            expr_name = args.resume
            tune.run(ppo.PPOTrainer, stop=stop_dict, local_dir=args.output_dir,
                     checkpoint_freq=args.checkpoint_freq, checkpoint_at_end=True, name=expr_name,
                     resume=True)
        else:
            # Setup experiment
            expr_name, config = experiment_setup(args=args)

            hpo_config = build_hpo_config(config, args)

            tune.run(ppo.PPOTrainer, config=config, stop=stop_dict, local_dir=args.output_dir,
                     checkpoint_freq=args.checkpoint_freq, checkpoint_at_end=True, name=expr_name,
                     restore=args.restore, callbacks=[TBXLoggerCallback()], **hpo_config)
    else:
        # Setup experiment
        expr_name, config = experiment_setup(args=args)

        # Run training in a single process for debugging
        config["num_workers"] = 0
        trainer = ppo.PPOTrainer(config=config)
        for i in range(args.stop_iteration):
            print(trainer.train())


if __name__ == "__main__":
    # Workaround for YAML not dumping ABCMeta objects
    # TODO: See if there is a better way to fix this
    from yaml.representer import Representer
    from abc import ABCMeta
    Representer.add_representer(ABCMeta, Representer.represent_name)

    parsed_args = get_args()

    main(parsed_args)
