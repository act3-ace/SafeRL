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

from rejoin_rta.environments.rejoin_env import DubinsRejoin

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='./output')

args = parser.parse_args()

expr_name =  datetime.now().strftime("expr_%Y%m%d_%H%M%S")
output_dir = os.path.join(args.output_dir, expr_name)

def run_rollout(agent, env_config):
    # instantiate env class
    env = DubinsRejoin(env_config)

    info_histroy = []
    obs_history = []
    reward_history = []
    action_history = []

    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        info_histroy.append(info)
        obs_history.append(obs)
        reward_history.append(reward)
        action_history.append(action)

    rollout_data = {
        'episode_reward': episode_reward,
        'info_history': info_histroy,
        'obs_history': obs_history,
        'reward_history': reward_history,
        'action_history': action_history
    }

    return rollout_data

ray.init(num_gpus=0)
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 6
config['_fake_gpus'] = True
config['seed'] = 0

rollout_history = []

reward_config = {
    'time_decay': -0.01,
    'failure_timeout': -1,
    'failure_crash': -1,
    'failure_distance': -1,
    'success': 1,
    'rejoin_timestep': 0.1,
    'rejoin_first_time': 0.25,
    'dist_change': -0.00001,
}

rejoin_config = {
    'reward': reward_config,
    'init': {
        'wingman': {
            'x': [-4000, 4000],
            'y': [-4000, 4000],
            'theta': [0, 2*math.pi],
            'velocity': [10, 100]
        },
        'lead': {
            'x': [-4000, 4000],
            'y': [-4000, 4000],
            'theta': [0, 2*math.pi],
            'velocity': [40, 60]
        },
    },
    'obs' : {
        # 'mode': 'rect',
        # 'reference': 'global',
        'mode': 'polar',
        'reference': 'wingman',
    },
    'rejoin_region' : {
        'type': 'circle',
        'range':500,
        'aspect_angle': 60,
        'radius':150,
    },
    'constraints':{
        'safety_margin': {
            'aircraft': 100
        },
        'timeout': 1000,
        'max_goal_distance': 40000,
        'success': {
            'rejoin_time': 20,
        },
    },
    'verbose':False,
}

config['env_config'] = rejoin_config
config['env'] = DubinsRejoin

stop_dict = {
    'training_iteration': 200,
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

tune.run(ppo.PPOTrainer, config=config, stop=stop_dict, local_dir=args.output_dir, checkpoint_freq=2, checkpoint_at_end=True, name=expr_name)
