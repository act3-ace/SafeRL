import numpy as np
import os
import argparse
import pickle
import jsonlines
import tqdm

import ray
import ray.rllib.agents.ppo as ppo

from saferl.aerospace.tasks.rejoin.task import DubinsRejoin
from saferl.aerospace.tasks.docking.task import DockingEnv
from saferl.environment.utils import jsonify, is_jsonable


class InvalidExperimentDirStructure(Exception):
    pass


def get_args():
    # function to process script args
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default="", help="the path to the experiment directory", required=True)
    parser.add_argument('--ckpt_num', type=int, default=None, help="specify a checkpoint to load")
    parser.add_argument('--seed', type=int, default=None, help="the seed used to initialize evaluation environment")
    parser.add_argument('--explore', type=bool, default=False, help="True for off-policy evaluation")
    parser.add_argument('--output_dir', type=str, default=None, help="Full path of directory to write evaluation logs")
    parser.add_argument('--num_rollouts', type=int, default=10, help="Number of randomly initialized episodes to evaluate")

    return parser.parse_args()


def run_rollouts(agent, env, log_dir, num_rollouts=1):
    """
    A function to coordinate policy evaluation via RLLib API.

    Parameters
    ----------
    agent : ray.rllib.agents.trainer_template.PPO
        The trained agent which will be evaluated.
    env : BaseEnv
        The environment in which the agent will act.
    log_dir : str
        The path to the output directory in which evaluation logs will be run.
    num_rollouts : int
        The number of randomly initialized episodes conducted to evaluate the agent on.
    """
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


def verify_experiment_dir(expr_dir_path):
    """
    A function to ensure passed path points to experiment run directory (as opposed to the parent directory).

    Parameters
    ----------
    expr_dir_path : str
        The full path to the experiment directory as received from arguments

    Returns
    -------
    The full path to the experiment run directory (containing the params.pkl file).
    """

    params_found = False
    params_dir_path = None
    if not os.path.isfile(os.path.join(expr_dir_path, 'params.pkl')):
        # if params.pkl not in dir (given path is experiment root dir), check its child dirs
        for child_dir_name in next(os.walk(expr_dir_path))[1]:
            child_dir_path = os.path.join(expr_dir_path, child_dir_name)
            if os.path.isfile(os.path.join(child_dir_path, 'params.pkl')):
                if params_found:
                    # there should only be one dir with params.pkl file
                    raise InvalidExperimentDirStructure("More than one params.pkl file found!")

                params_found = True
                params_dir_path = child_dir_path

        if params_dir_path is not None:
            expr_dir_path = params_dir_path
        else:
            raise InvalidExperimentDirStructure("No params.pkl file found!")

    return expr_dir_path


def find_checkpoint_dir(ckpt_num):
    """
    A function to locate the checkpoint and trailing identifying numbers of a checkpoint file.

    Parameters
    ----------
    ckpt_num : int
        The identifying checkpoint number which the user wishes to evaluate.

    Returns
    -------
    ckpt_num : int
        The specified ckpt number (or the number of the latest saved checkpoint, if no ckpt_num was specifed).
    ckpt_num_str : str
        The leading numbers of the checkpoint directory corresponding to the desired checkpoint.
    """
    ckpt_num_str = None

    # find checkpoint dir
    if ckpt_num is not None:
        # ckpt specified
        for ckpt_dir_name in next(os.walk(expr_dir_path))[1]:
            if "_" in ckpt_dir_name:
                ckpt_num_str = ckpt_dir_name.split("_")[1]
                if ckpt_num == int(ckpt_num_str):
                    # handling nuance of ckpt dirs formatting numbers with leading zeroes
                    ckpt_num = int(ckpt_num_str)
                    break
    else:
        ckpt_num = -1
        for ckpt_dir_name in next(os.walk(expr_dir_path))[1]:
            if "_" in ckpt_dir_name:
                file_num = ckpt_dir_name.split("_")[1]
                if ckpt_num < int(file_num):
                    ckpt_num = int(file_num)
                    ckpt_num_str = file_num

    return ckpt_num, ckpt_num_str


## process args
args = get_args()

# assume full path passed in
expr_dir_path = args.dir

# verify experiment run dir
expr_dir_path = verify_experiment_dir(expr_dir_path)

# get checkpoint num
ckpt_num, ckpt_num_str = find_checkpoint_dir(args.ckpt_num)

# set paths
eval_dir_path = os.path.join(expr_dir_path, 'eval')
ckpt_eval_dir_path = os.path.join(eval_dir_path, 'ckpt_{}'.format(ckpt_num))

ray_config_path = os.path.join(expr_dir_path, 'params.pkl')
ckpt_dir = 'checkpoint_{}'.format(ckpt_num_str)
ckpt_filename = 'checkpoint-{}'.format(ckpt_num)
ckpt_path = os.path.join(expr_dir_path, ckpt_dir, ckpt_filename)

# user specified output
if args.output_dir is not None:
    eval_dir_path = args.output_dir
    ckpt_eval_dir_path = os.path.join(eval_dir_path, 'ckpt_{}'.format(ckpt_num))

# make directories
os.makedirs(eval_dir_path, exist_ok=True)
os.makedirs(ckpt_eval_dir_path, exist_ok=True)

## load checkpoint
with open(ray_config_path, 'rb') as ray_config_f:
    ray_config = pickle.load(ray_config_f)

ray.init()
env_config = ray_config['env_config']
agent = ppo.PPOTrainer(config=ray_config, env=ray_config['env'])
agent.restore(ckpt_path)
env = ray_config['env'](config=env_config)

seed = args.seed if args.seed is not None else ray_config['seed']
env.seed(seed)
agent.get_policy().config['explore'] = args.explore

## run inference episodes and log results
run_rollouts(agent, env, ckpt_eval_dir_path + "/eval.log", num_rollouts=args.num_rollouts)
