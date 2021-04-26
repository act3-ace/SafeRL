import numpy as np
import os
import argparse
import pickle
# import pickle5 as pickle
import tqdm
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# import tensorflow.keras as keras

# from scipy.io import savemat

import ray

import ray.rllib.agents.ppo as ppo

from saferl.aerospace.tasks.rejoin.task import DubinsRejoin
from saferl.aerospace.tasks.docking.task import DockingEnv
# from saferl.environment.utils import numpy_to_matlab_txt


def get_args():
    # method to process script args
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default="", help="the path to the experiment directory", required=True)
    parser.add_argument('--ckpt_num', type=int, default=None, help="specify a checkpoint to load")
    parser.add_argument('--seed', type=int, default=None, help="the seed ")
    parser.add_argument('--explore', type=bool, default=False, help="True if off-policy evaluation desired")    #?

    return parser.parse_args()

def run_rollouts(agent, env, num_rollouts=1):

    rollout_seq = []

    for i in tqdm.tqdm(range(num_rollouts)):

        info_history = []
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

            info_history.append(info)
            obs_history.append(obs)
            reward_history.append(reward)
            action_history.append(action)

        rollout_data = {
            'episode_reward': episode_reward,
            'info_history': info_history,
            'obs_history': obs_history,
            'reward_history': reward_history,
            'action_history': action_history
        }

        rollout_seq.append(rollout_data)

    return rollout_seq

# process args
# args = get_args()
# expr_dir_path = args.dir #TODO: ensure exists, complete path
# ckpt_num = args.ckpt_num
# if ckpt_num is None:


# get paths
ckpt_num = 200
expr_dir_path = '/home/john/AFRL/Dubins/have-deepsky/scripts/output/expr_20210426_093952/PPO_DockingEnv_e2f21_00000_0_2021-04-26_09-39-55'
# expr_dir_path = args.dir
eval_dir_path = os.path.join(expr_dir_path, 'eval')
ckpt_eval_dir_path = os.path.join(eval_dir_path, 'ckpt_{}'.format(ckpt_num))

ray_config_path = os.path.join(expr_dir_path, 'params.pkl')
ckpt_dir_name = 'checkpoint_{}'.format(ckpt_num)
ckpt_filename = 'checkpoint-{}'.format(ckpt_num)
ckpt_path = os.path.join(expr_dir_path, ckpt_dir_name, ckpt_filename)

## load checkpoint
with open(ray_config_path, 'rb') as ray_config_f:
    ray_config = pickle.load(ray_config_f)

ray.init()

env_config = ray_config['env_config']
# ray_config['callbacks'] = ppo.DEFAULT_CONFIG['callbacks']       #TODO: add new logging callback?

agent = ppo.PPOTrainer(config=ray_config, env=ray_config['env'])
agent.restore(ckpt_path)

env = ray_config['env'](config=env_config)
# TODO: set seed from args
env.seed(ray_config['seed'])

# TODO: set explore from args
agent.get_policy().config['explore'] = False

# TODO: pass in init ranges from args?

# run inference episodes
results = run_rollouts(agent, env, num_rollouts=1)

# output results (tensorboard & logging)


1+1




# Backlog #

# log performance (expr/eval/ckpt_<num>) w/o Tune
# run eval and log performance with Tune (callback and tensorboard logs)
# parse args intelligently
# robust imports of Envs?
# init ranges from args? env from args?


# Done #

# load model and env
# run eval with std RLLib python api
