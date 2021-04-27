import numpy as np
import os
import argparse
import pickle
import jsonlines
import tqdm
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# import tensorflow.keras as keras

# from scipy.io import savemat

import ray

import ray.rllib.agents.ppo as ppo

from saferl.aerospace.tasks.rejoin.task import DubinsRejoin
from saferl.aerospace.tasks.docking.task import DockingEnv
from saferl.environment.utils import jsonify, is_jsonable


def get_args():
    # method to process script args
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default="", help="the path to the experiment directory", required=True)
    parser.add_argument('--ckpt_num', type=int, default=None, help="specify a checkpoint to load")
    parser.add_argument('--seed', type=int, default=None, help="the seed ")
    parser.add_argument('--explore', type=bool, default=False, help="True if off-policy evaluation desired")

    return parser.parse_args()

def run_rollouts(agent, env, log_dir, num_rollouts=1):

    for i in tqdm.tqdm(range(num_rollouts)):
        # run until episode ends
        episode_reward = 0
        done = False
        obs = env.reset()
        step_num = 0

        with jsonlines.open(log_dir, "a") as writer:
            while not done:
                # progress environment state
                action = agent.compute_action(obs)
                obs, reward, done, info = env.step(action)
                step_num += 1
                episode_reward += reward

                # store log contents in state
                state = {}
                if is_jsonable(info) == True:
                    state["info"] = info
                else:
                    state["info"] = jsonify(info)
                state["actions"] = [float(i) for i in action]
                state["obs"] = obs.tolist()
                state["rollout_num"] = i
                state["step_number"] = step_num
                state["episode_reward"] = episode_reward

                # write state to file
                writer.write(state)

# process args
# args = get_args()
# expr_dir_path = args.dir #TODO: ensure exists, complete path
# ckpt_num = args.ckpt_num
# if ckpt_num is None:


# get paths
ckpt_num_str = "000200"
ckpt_num = 200
expr_dir_path = '/home/john/AFRL/Dubins/have-deepsky/scripts/output/expr_20210427_141133/PPO_DockingEnv_0236e_00000_0_2021-04-27_14-11-38'
# expr_dir_path = args.dir
eval_dir_path = os.path.join(expr_dir_path, 'eval')
ckpt_eval_dir_path = os.path.join(eval_dir_path, 'ckpt_{}'.format(ckpt_num))

ray_config_path = os.path.join(expr_dir_path, 'params.pkl')
ckpt_dir_name = 'checkpoint_{}'.format(ckpt_num_str)
ckpt_filename = 'checkpoint-{}'.format(ckpt_num)
ckpt_path = os.path.join(expr_dir_path, ckpt_dir_name, ckpt_filename)

# make directories
os.makedirs(eval_dir_path, exist_ok=True)
os.makedirs(ckpt_eval_dir_path, exist_ok=True)

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

# run inference episodes and log results
run_rollouts(agent, env, ckpt_eval_dir_path + "/eval.log", num_rollouts=3)

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
