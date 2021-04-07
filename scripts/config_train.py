import argparse
import os

import yaml

from datetime import datetime

import ray
from ray import tune

import ray.rllib.agents.ppo as ppo

from saferl.environment import build_callbacks_caller, EpisodeOutcomeCallback, FailureCodeCallback, \
                                        RewardComponentsCallback, LoggingCallback, LogContents

from saferl import saferl_lookup

from saferl.environment.utils import parse_env_config


# Training defaults

OUTPUT_DIR = './output'
NUM_LOGGING_WORKERS = 1
LOGGING_INTERVAL = 10  # log every 10th episode
CONTENTS = (LogContents.VERBOSE,)
CUDA_VISIBLE_DEVICES = "-1"
NUM_GPUS = 0
NUM_WORKERS = 6
FAKE_GPUS = True
SEED = 0
STOP_ITERATION = 200
CHECKPOINT_FREQUENCY = 25

DEBUG = False


def get_args():
    parser = argparse.ArgumentParser()

    # Add parser arguments
    parser.add_argument('--config', type=str)
    parser.add_argument('--debug', default=False, action="store_true")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--logging_workers', type=int, default=NUM_LOGGING_WORKERS)
    parser.add_argument('--log_interval', type=int, default=LOGGING_INTERVAL)
    parser.add_argument('--checkpoint_freq', type=int, default=CHECKPOINT_FREQUENCY)
    parser.add_argument('--cuda_visible', type=str, default=CUDA_VISIBLE_DEVICES)
    parser.add_argument('--gpus', type=int, default=NUM_GPUS)
    parser.add_argument('--workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--fake_gpus', default=False, action="store_true")
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--stop_iteration', type=int, default=STOP_ITERATION)

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

    # TODO: allow choice of default config

    # Setup default PPO config
    config = ppo.DEFAULT_CONFIG.copy()
    config["num_gpus"] = args.gpus
    config["num_workers"] = args.workers
    config['_fake_gpus'] = args.fake_gpus
    config['seed'] = args.seed
    config['callbacks'] = build_callbacks_caller([EpisodeOutcomeCallback(),
                                                  FailureCodeCallback(),
                                                  RewardComponentsCallback(),
                                                  LoggingCallback(num_logging_workers=args.logging_workers,
                                                                  episode_log_interval=args.log_interval,
                                                                  contents=CONTENTS)])

    # Setup custom config
    env, env_config = parse_env_config(config_yaml=args.config, lookup=saferl_lookup)
    config['env'] = env
    config['env_config'] = env_config

    # Save experiment params
    with open(args_yaml_filepath, 'w') as args_yaml_file:
        arg_dict = vars(args)
        yaml.dump(arg_dict, args_yaml_file)
    with open(ray_config_yaml_filepath, 'w') as ray_config_yaml_file:
        yaml.dump(config, ray_config_yaml_file)

    # Initialize stop dict
    stop_dict = {
        'training_iteration': args.stop_iteration,
    }

    return expr_name, config, stop_dict


def main(args):

    # Setup experiment
    expr_name, config, stop_dict = experiment_setup(args=args)

    # Run training
    if not args.debug:
        tune.run(ppo.PPOTrainer, config=config, stop=stop_dict, local_dir=args.output_dir,
                 checkpoint_freq=args.checkpoint_freq, checkpoint_at_end=True, name=expr_name)
    else:
        # Run training in a single process for debugging
        config["num_workers"] = 0
        trainer = ppo.PPOTrainer(config=config)
        while True:
            print(trainer.train())


if __name__ == "__main__":
    # Workaround for YAML not dumping ABCMeta objects
    # TODO: See if there is a better way to fix this
    from yaml.representer import Representer
    from abc import ABCMeta
    Representer.add_representer(ABCMeta, Representer.represent_name)

    parsed_args = get_args()

    main(parsed_args)
