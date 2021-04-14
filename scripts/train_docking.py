import argparse
import os
import yaml

from datetime import datetime

import ray
from ray import tune

import ray.rllib.agents.ppo as ppo

from saferl.environment.callbacks import build_callbacks_caller, EpisodeOutcomeCallback, FailureCodeCallback, \
                                        RewardComponentsCallback, LoggingCallback

from saferl.aerospace.tasks import *


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='./output')

args = parser.parse_args()

expr_name =  datetime.now().strftime("expr_%Y%m%d_%H%M%S")
output_dir = os.path.join(args.output_dir, expr_name)

# set logging verbosity options
num_logging_workers = 2
logging_schedule = 10       # log every 10th episode

ray.init(num_gpus=0)
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 6
config['_fake_gpus'] = True
config['seed'] = 0
config['callbacks'] = build_callbacks_caller([
    EpisodeOutcomeCallback(), 
    FailureCodeCallback(), 
    RewardComponentsCallback(),
    LoggingCallback(num_logging_workers, logging_schedule),
    ])

rollout_history = []

# ------ Define reward configuration ------

reward_processors = [
    TimeRewardProcessor,
    DistanceChangeRewardProcessor,
    FailureRewardProcessor,
    SuccessRewardProcessor
]

reward_processors_3d = [
    TimeRewardProcessor,
    DistanceChangeZRewardProcessor,
    FailureRewardProcessor,
    SuccessRewardProcessor
]

reward_config = {
    'processors': reward_processors,
    'time_decay': -0.01,
    'failure': {
        'timeout': -1,
        'crash': -1,
        'distance': -1,
    },
    'success': 1,
    'dist_change': -0.0001,
}

reward_config_3d = {
    'processors': reward_processors_3d,
    'time_decay': -0.001,
    'failure': {
        'timeout': -1,
        'crash': -1,
        'distance': -10,
    },
    'success': 100,
    'dist_change': -0.001,
    'dist_z_change': -0.01,
}

# ------ Define status configuration ------

status_processors = [
    DockingStatusProcessor,
    DockingDistanceStatusProcessor,
    FailureStatusProcessor,
    SuccessStatusProcessor
]

status_config = {
    'processors': status_processors,
    'timeout': 1000,
    'max_goal_distance': 40000,
}

# ------ Define observation configuration ------

observation_processors = [
    DockingObservationProcessor
]

observation_config = {
    'processors': observation_processors,
    'mode': '2d'
}

observation_config_3d = {
    'processors': observation_processors,
    'mode': '3d'
}

# ------ Define environment configuration ------

env_config = {
    'reward': reward_config,
    'mode': '2d',
    'init': {
        'deputy': {
            'x': [-2000, 2000],
            'y': [-2000, 2000],
            'x_dot': 0,
            'y_dot': 0,
        },
        'chief': {
            'x': 0,
            'y': 0,
            'x_dot': 0,
            'y_dot': 0,
        },
    },
    'agent':{
        'controller':{
            'actuators': [
                {
                    'name': 'thrust_x',
                    'space': 'discrete',
                    'points': 11,
                    'bounds': [-10, 10]
                },
                {
                    'name': 'thrust_y',
                    'space': 'discrete',
                    'points': 11,
                    'bounds': [-10, 10]
                },
            ]
        },
    },
    'observation': observation_config,
    'docking_region': {
        'type': 'circle',
        'radius': 20,
    },
    'status': status_config,
    'verbose': False,
}

env_config3d = {
    'reward': reward_config_3d,
    'mode': '3d',
    'init': {
        'deputy': {
            'x': 1000,
            'y': 1000,
            'z': [-2000, 2000],
            'x_dot': 0,
            'y_dot': 0,
            'z_dot': 0,
        },
        'chief': {
            'x': 0,
            'y': 0,
            'z': 0,
            'x_dot': 0,
            'y_dot': 0,
            'z_dot': 0,
        },
    },
    'agent':{
        'controller':{
            'actuators': [
                {
                    'name': 'thrust_x',
                    'space': 'discrete',
                    'points': 11,
                    'bounds': [-10, 10]
                },
                {
                    'name': 'thrust_y',
                    'space': 'discrete',
                    'points': 11,
                    'bounds': [-10, 10]
                },
                {
                    'name': 'thrust_z',
                    'space': 'discrete',
                    'points': 11,
                    'bounds': [-10,10]
                },
            ],
        },
    },
    'observations': observation_config_3d,
    'docking_region': {
        'type': 'cylinder',
        'params': {
            'radius': 300,
            'height': 600,
        }
    },
    'status': status_config,
    'verbose': False,
}

config['env_config'] = env_config
config['env'] = DockingEnv

stop_dict = {
    'training_iteration': 500,
}

if __name__ == "__main__":
    # Workaround for YAML not dumping ABCMeta objects
    # TODO: See if there is a better way to fix this
    from yaml.representer import Representer
    from abc import ABCMeta
    Representer.add_representer(ABCMeta, Representer.represent_name)

    DEBUG = False

    # create output dir and save experiment params
    os.makedirs(output_dir, exist_ok=True)
    args_yaml_filepath = os.path.join(output_dir, 'script_args.yaml')
    ray_config_yaml_filepath = os.path.join(output_dir, 'ray_config.yaml')
    with open(args_yaml_filepath, 'w') as args_yaml_file:
        arg_dict = vars(args)
        yaml.dump(arg_dict, args_yaml_file)
    with open(ray_config_yaml_filepath, 'w') as ray_config_yaml_file:
        yaml.dump(config, ray_config_yaml_file)

    if not DEBUG:
        tune.run(ppo.PPOTrainer, config=config, stop=stop_dict, local_dir=args.output_dir, checkpoint_freq=25, checkpoint_at_end=True, name=expr_name)
    else:
        # Run training in a single process for debugging
        config["num_workers"] = 0
        trainer = ppo.PPOTrainer(config=config)
        while True:
            print(trainer.train())
