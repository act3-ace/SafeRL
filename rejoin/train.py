import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import yaml

from datetime import datetime

import pickle

import ray
from ray import tune

import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

from rejoin_rta.environments.rejoin_env import DubinsRejoin

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='./output')

args = parser.parse_args()

output_dir = os.path.join(args.output_dir, datetime.now().strftime("expr_%Y%m%d_%H%M%S") )

config={
    'env': DubinsRejoin,
    'env_config':{},
}

stop = {
    "training_iteration": 50,
    "timesteps_total": 1000000,
}

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

# results = tune.run("PPO", config=config, stop=stop)

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
# config["eager"] = False

rollout_history = []

reward_config = {
    'time_decay': -0.1,
    'failure_timeout': -10,
    'failure_crash': -1000,
    'failure_distance': -100,
    'success': 1000,
    'rejoin_timestep': 1,
    'rejoin_first_time': 10,
    'dist_change': -1/100,
}

config['env_config'] = {'reward_config':reward_config}

trainer = ppo.PPOTrainer(config=config, env=DubinsRejoin)

eixt()

# create output dir and save experiment params
os.makedirs(output_dir)
args_yaml_filepath = os.path.join(output_dir, 'script_args.yaml')
ray_config_yaml_filepath = os.path.join(output_dir, 'ray_config.yaml')
with open(args_yaml_filepath, 'w') as args_yaml_file:
    arg_dict = vars(args)
    yaml.dump(arg_dict, args_yaml_file)
with open(ray_config_yaml_filepath, 'w') as ray_config_yaml_file:
    yaml.dump(config, ray_config_yaml_file)

rollout_history.append(run_rollout(trainer, config['env_config']))
for i in range(200):
    result = trainer.train()
    rollout_history.append(run_rollout(trainer, config['env_config']))
    print(pretty_print(result))

save_data = {
    'config':config,
    'rollout_history':rollout_history
}

rollout_history_filepath = os.path.join(output_dir, "rollout_history.pickle")

pickle.dump( save_data, open( rollout_history_filepath, "wb" ) )