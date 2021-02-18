import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import yaml
import math

from datetime import datetime

import pickle

import ray
from ray import tune

import ray.rllib.agents.ppo as ppo
from ray.tune.logger import JsonLogger

from rejoin_rta.environments.docking_env import DockingEnv, DockingObservationProcessor, DockingRewardProcessor, DockingRewardProcessor3D, DockingConstraintProcessor
from rejoin_rta.utils.callbacks import build_callbacks_caller, EpisodeOutcomeCallback, FailureCodeCallback, RewardComponentsCallback, LoggingCallback

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
    # LoggingCallback(num_logging_workers, logging_schedule),
    ])

rollout_history = []

reward_config = {
    'processor': DockingRewardProcessor,
    'time_decay': -0.001,
    'failure': {
        'timeout': -1,
        'crash': -1,
        'distance': -1,
    },
    'success': 1,
    'dist_change': -0.0001,
}

reward_config_3d = {
    'processor': DockingRewardProcessor3D,
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

env_config = {
    'reward': reward_config,
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
            'type': 'agent',
            'actuators': {
                'thrust_x': {
                    'space': 'discrete',
                    'points': 11,
                    'bounds': [-10, 10]
                },
                'thrust_y': {
                    'space': 'discrete',
                    'points': 11,
                    'bounds': [-10, 10]
                },
            },
        },
    },
    'obs' : {
        'processor': DockingObservationProcessor,
        'mode': '2d'
    },
    'docking_region' : {
        'type': 'circle',
        'radius': 20,
    },
    'constraints':{
        'processor': DockingConstraintProcessor,
        'timeout': 1000,
        'max_goal_distance': 40000,
    },
    'verbose':False,
}

env_config3d = {
    'reward': reward_config_3d,
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
            'type': 'agent',
            'actuators': {
                # 'thrust_x': {
                #     'space': 'discrete',
                #     'points': 11,
                #     'bounds': [-10, 10]
                # },
                # 'thrust_y': {
                #     'space': 'discrete',
                #     'points': 11,
                #     'bounds': [-10, 10]
                # },
                # 'thrust_z': {
                #     'space': 'discrete',
                #     'points': 11,
                #     'bounds': [-10,10]
                # },
                'thrust_x': {
                    'space': 'continuous',
                    'bounds': [-10, 10]
                },
                'thrust_y': {
                    'space': 'continuous',
                    'bounds': [-10, 10]
                },
                'thrust_z': {
                    'space': 'continuous',
                    'bounds': [-10,10]
                },
            },
        },
    },
    'obs' : {
        'processor': DockingObservationProcessor,
        'mode': '3d'
    },
    'docking_region' : {
        'type': 'cylinder',
        'params': {
            'radius': 300,
            'height': 600,
        }
    },
    'constraints':{
        'processor': DockingConstraintProcessor,
        'timeout': 1000,
        'max_goal_distance': 10000,
    },
    'verbose':False,
}

config['env_config'] = env_config
config['env'] = DockingEnv

stop_dict = {
    'training_iteration': 500,
}

# create output dir and save experiment params
os.makedirs(output_dir, exist_ok=True)
args_yaml_filepath = os.path.join(output_dir, 'script_args.yaml')
ray_config_yaml_filepath = os.path.join(output_dir, 'ray_config.yaml')
with open(args_yaml_filepath, 'w') as args_yaml_file:
    arg_dict = vars(args)
    yaml.dump(arg_dict, args_yaml_file)
with open(ray_config_yaml_filepath, 'w') as ray_config_yaml_file:
    yaml.dump(config, ray_config_yaml_file)

tune.run(ppo.PPOTrainer, config=config, stop=stop_dict, local_dir=args.output_dir, checkpoint_freq=25, checkpoint_at_end=True, name=expr_name)
