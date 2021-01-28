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

from rejoin_rta.environments.docking_env import DockingEnv
from vis_saved_rollouts import animate_trajectories_docking, process_rollout_data

def run_rollouts(agent, env_config, num_rollouts=1):
    # instantiate env class
    env = DockingEnv(env_config)

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

def compare_rollouts(expr_dir, ckpt_nums):
    rollout_seq_set = []

    ray_config_path = os.path.join(expr_dir, 'params.pkl')
    with open(ray_config_path, 'rb') as ray_config_f:
        ray_config = pickle.load(ray_config_f)
    env_config = ray_config['env_config']

    ckpt_dir_name = 'checkpoints_' + 'v'.join([str(a) for a in ckpt_nums])

    output_dir = os.path.join(expr_dir, 'rollouts', ckpt_dir_name)
    output_anim_dir = os.path.join(output_dir, 'animations')
    output_anim_single_dir = os.path.join(output_anim_dir, 'single_rollouts')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_anim_dir, exist_ok=True)
    os.makedirs(output_anim_single_dir, exist_ok=True)

    for ckpt_num in ckpt_nums:
        ckpt_dir_name = 'checkpoint_{}'.format(ckpt_num)
        rollout_dir = os.path.join(expr_dir, 'rollouts', ckpt_dir_name)
        rollout_history_filepath = os.path.join(rollout_dir, "rollout_history.pickle")
        rollout_seq = pickle.load( open( rollout_history_filepath, "rb" ) )
        rollout_seq_set.append(rollout_seq)

    actuator_config = ray_config['env_config']['agent']['controller']['actuators']

    for i in tqdm.tqdm(range(len(rollout_seq_set[0]))):
        
        rollout_seq = [ rollout_seq[i] for rollout_seq in rollout_seq_set]

        animate_trajectories_docking(rollout_seq, os.path.join(output_anim_single_dir,'rollout_{:03d}.mp4'.format(i)), anim_rate=1, plot_docking_region=True, sq_axis=True, plot_estimated_trajectory=True, frame_interval=66,
            plot_actuators=True, actuator_config=actuator_config, trail_length=20)



if __name__ == '__main__':
    expr_dir = 'output/expr_20210127_170415/PPO_DockingEnv_beafd_00000_0_2021-01-27_17-04-17'
    ckpt_num = 500
    ckpt_nums = [200, 500]
    only_failures = False
    load_rollouts = False
    compare = True

    if compare:
        compare_rollouts(expr_dir, ckpt_nums)
        exit()

    ray_config_path = os.path.join(expr_dir, 'params.pkl')

    ckpt_dir_name = 'checkpoint_{}'.format(ckpt_num)
    ckpt_filename = 'checkpoint-{}'.format(ckpt_num)
    ckpt_path = os.path.join(expr_dir, ckpt_dir_name, ckpt_filename)

    with open(ray_config_path, 'rb') as ray_config_f:
        ray_config = pickle.load(ray_config_f)

    ray.init()

    env_config = ray_config['env_config']
    # ray_config['callbacks'] = ppo.DEFAULT_CONFIG['callbacks']

    agent = ppo.PPOTrainer(config=ray_config, env=DockingEnv)
    agent.restore(ckpt_path)

    

    output_dir = os.path.join(expr_dir, 'rollouts', ckpt_dir_name)
    output_anim_dir = os.path.join(output_dir, 'animations')
    if only_failures:
        output_anim_single_dir = os.path.join(output_anim_dir, 'failures')
    else:
        output_anim_single_dir = os.path.join(output_anim_dir, 'single_rollouts')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_anim_dir, exist_ok=True)
    os.makedirs(output_anim_single_dir, exist_ok=True)

    rollout_history_filepath = os.path.join(output_dir, "rollout_history.pickle")

    if not load_rollouts:

        print('running rollouts')
        rollout_seq = run_rollouts(agent, env_config, num_rollouts=20)
        print('finished running rollouts')

        pickle.dump( rollout_seq, open( rollout_history_filepath, "wb" ) )

    else:
        rollout_seq = pickle.load( open( rollout_history_filepath, "rb" ) )

    failure_rollout_seq = []

    # actuator_config = { 'thrust_x':{ 'bounds': [-100, 100] }, 'thrust_y':{ 'bounds': [-100, 100] }  }
    actuator_config = ray_config['env_config']['agent']['controller']['actuators']

    print('animating individual rollouts')
    for i, rollout in tqdm.tqdm(enumerate(rollout_seq), total=len(rollout_seq)):
        if rollout['info_history'][-1]['failure']:
            outcome_str = rollout['info_history'][-1]['failure']
            failure_rollout_seq.append(rollout)
        elif rollout['info_history'][-1]['success']:
            outcome_str = 'success'
            if only_failures:
                continue
        
        animate_trajectories_docking([rollout], os.path.join(output_anim_single_dir,'rollout_{:03d}_{}.mp4'.format(i, outcome_str)), anim_rate=1, plot_docking_region=True, sq_axis=True, plot_estimated_trajectory=True, frame_interval=66,
            plot_actuators=True, actuator_config=actuator_config, trail_length=20)

    print('animating all rollouts')

    if not only_failures:
        animate_trajectories_docking(rollout_seq, os.path.join(output_anim_dir,'all_trajectories.mp4'), anim_rate=1, plot_docking_region=True, sq_axis=True, plot_estimated_trajectory=False, frame_interval=66,
                plot_actuators=True, actuator_config=actuator_config, trail_length=20)
    
    # if len(failure_trajectory_data) > 0:
    #     animate_trajectories(failure_trajectory_data, os.path.join(output_anim_dir,'all_failure_trajectories.mp4'), plot_rejoin_region=True, rejoin_color_type='match')
