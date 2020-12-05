import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
import yaml
import math
import tqdm

from datetime import datetime

import pickle

import ray
from ray import tune

import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

from rejoin_rta.environments.rejoin_env import DubinsRejoin
from vis_saved_rollouts import animate_trajectories, process_rollout_data

def run_rollouts(agent, env_config, num_rollouts=1):
    # instantiate env class
    env = DubinsRejoin(env_config)

    rollout_seq = []

    for i in tqdm.tqdm(range(num_rollouts)):

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

        rollout_seq.append(rollout_data)

    return rollout_seq

if __name__ == '__main__':
    expr_dir = 'output/expr_20201202_155655'
    ckpt_num = 200

    ray_config_path = os.path.join(expr_dir, 'ray_config.yaml')

    ckpt_dir_name = 'checkpoint_{}'.format(ckpt_num)
    ckpt_filename = 'checkpoint-{}'.format(ckpt_num)
    ckpt_path = os.path.join(expr_dir, 'ckpt', ckpt_dir_name, ckpt_filename)

    with open(ray_config_path, 'r') as ray_config_f:
        ray_config = yaml.load(ray_config_f)

    ray.init()

    env_config = ray_config['env_config']
    ray_config['callbacks'] = ppo.DEFAULT_CONFIG['callbacks']

    agent = ppo.PPOTrainer(config=ray_config, env=DubinsRejoin)
    agent.restore(ckpt_path)

    print('running rollouts')
    rollout_seq = run_rollouts(agent, env_config, num_rollouts=20)
    print('finished running rollouts')

    output_dir = os.path.join(expr_dir, 'rollouts', ckpt_dir_name)
    output_anim_dir = os.path.join(output_dir, 'animations')
    output_anim_single_dir = os.path.join(output_anim_dir, 'single_rollouts')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_anim_dir, exist_ok=True)
    os.makedirs(output_anim_single_dir, exist_ok=True)

    rollout_history_filepath = os.path.join(output_dir, "rollout_history.pickle")
    pickle.dump( rollout_seq, open( rollout_history_filepath, "wb" ) )

    print('processing rollout trajectories')
    trajectory_data = process_rollout_data(rollout_seq)

    print('animating individual rollouts')
    for i in tqdm.tqdm(range(len(trajectory_data))):
        animate_trajectories([trajectory_data[i]], os.path.join(output_anim_single_dir,'rollout_{:03d}.mp4'.format(i)), anim_rate=1, plot_rejoin_region=True, plot_safety_region=True, sq_axis=True)

    print('animating all rollouts')
    animate_trajectories(trajectory_data, os.path.join(output_anim_dir,'all_trajectories.mp4'), plot_rejoin_region=True, rejoin_color_type='match')
