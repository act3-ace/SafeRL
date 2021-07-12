import argparse
import os
from distutils.util import strtobool
import yaml

from datetime import datetime

import ray
from ray import tune

import ray.rllib.agents.ppo as ppo

from saferl.environment.utils import YAMLParser, build_lookup
from saferl.environment.callbacks import build_callbacks_caller, EpisodeOutcomeCallback, FailureCodeCallback, \
                                        RewardComponentsCallback, LoggingCallback, LogContents


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
EVALUATION_SEED = 1
DEBUG = False
COMPLETE = False
ROLLOUT_FRAGMENT_LENGTH = None


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
    parser.add_argument('--gpus', type=int, default=NUM_GPUS, help="number of gpus used for training")
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, help="number of cpu workers used for training")
    parser.add_argument(
        '--envs_per_worker',
        type=int,
        default=NUM_ENVS_PER_WORKER,
        help="number of environments per cpu worker used for training"
    )
    parser.add_argument('--fake_gpus', default=False, action="store_true", help="use simulated gpus")
    parser.add_argument('--seed', type=int, default=SEED, help="set random seed")
    parser.add_argument('--stop_iteration', type=int, default=STOP_ITERATION, help="number of iterations to run")

    parser.add_argument('--complete_episodes', type=lambda x: strtobool(x), default=COMPLETE,
                        help="True if using complete episodes during training desired, "
                             "False if using truncated episodes")
    parser.add_argument('--rollout_fragment_length', type=int, default=ROLLOUT_FRAGMENT_LENGTH,
                        help="size of batches collected by each worker if truncated episodes")

    parser.add_argument('--evaluation_during_training', type=lambda x: strtobool(x), default=False,
                        help="True if intermittent evaluation of agent policy during training desired, False if not")
    parser.add_argument('--evaluation_interval', type=int, default=EVALUATION_INTERVAL,
                        help="number of episodes to run in between policy evaluations")
    parser.add_argument('--evaluation_num_episodes', type=int, default=EVALUATION_NUM_EPISODES,
                        help="number of evaluation episodes to run")
    parser.add_argument('--evaluation_num_workers', type=int, default=EVALUATION_NUM_WORKERS,
                        help="number of workers used to run evaluation episodes")
    parser.add_argument('--evaluation_seed', type=int, default=EVALUATION_SEED,
                        help="set random seed for evaluation episodes")
    parser.add_argument('--evaluation_exploration', type=lambda x: strtobool(x), default=False,
                        help="set exploration behavior for evaluation episodes")

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
    if args.envs_per_worker > 1:
        config["num_envs_per_worker"] = args.envs_per_worker
    config['callbacks'] = build_callbacks_caller([EpisodeOutcomeCallback(),
                                                  FailureCodeCallback(),
                                                  RewardComponentsCallback(),
                                                  LoggingCallback(num_logging_workers=args.logging_workers,
                                                                  episode_log_interval=args.log_interval,
                                                                  contents=CONTENTS)])

    config['batch_mode'] = "complete_episodes" if args.complete_episodes else "truncate_episodes"
    if args.rollout_fragment_length is not None:
        config['rollout_fragment_length'] = args.rollout_fragment_length

    # Setup custom config
    parser = YAMLParser(yaml_file=args.config, lookup=build_lookup())
    env, env_config = parser.parse_env()
    config['env'] = env
    config['env_config'] = env_config

    if args.evaluation_during_training:
        # set evaluation parameters
        config["evaluation_interval"] = args.evaluation_interval
        config["evaluation_num_episodes"] = args.evaluation_num_episodes
        config["evaluation_num_workers"] = args.evaluation_num_workers
        config["evaluation_config"] = {
            # override config for logging, tensorboard, etc
            "explore": args.evaluation_exploration,
            "seed": args.evaluation_seed,
            "callbacks": build_callbacks_caller([EpisodeOutcomeCallback(),
                                                 FailureCodeCallback(),
                                                 RewardComponentsCallback(),
                                                 LoggingCallback(num_logging_workers=args.evaluation_num_workers,
                                                                 contents=CONTENTS)])
        }

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
