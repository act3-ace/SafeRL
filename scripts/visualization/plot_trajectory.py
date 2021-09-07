"""
This module is responsible for consuming experiment directories and creating trajectory data from evaluation episodes.
This data is then used to create trajectory plots. This is useful for showing policy improvements throughout training
and creating figures for our public paper.
"""

# import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
import argparse
from glob import glob
import pickle5 as pickle
import ray
import ray.rllib.agents.ppo as ppo

from scripts.eval import run_rollouts, find_checkpoint_dir, verify_experiment_dir

DEFAULT_NUM_CKPTS = 5
DEFAULT_SEED = 33
DEFAULT_OUTPUT = "/figures/data"


def get_args():
    """
    A function to process script args.

    Returns
    -------
    argparse.Namespace
        Collection of command line arguments and their values
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default="", help="The full path to the experiment directory", required=True)
    parser.add_argument('--num_ckpts', type=int, default=DEFAULT_NUM_CKPTS, help="Number of checkpoints to plot")
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help="The seed used to initialize the evaluation environment")
    # add output dir

    return parser.parse_args()


def main():
    # collect experiment path
    args = get_args()
    expr_dir_path = args.dir
    num_ckpts = args.num_ckpts

    # locate checkpoints
    expr_dir_path = verify_experiment_dir(expr_dir_path)
    ckpt_dirs = glob(expr_dir_path + "/checkpoint_*")

    ckpts_processed = 0
    for ckpt_dir_name in ckpt_dirs:
        # iterate through checkpoints
        # assuming ckpt_dirs ordered from latest to earliest

        # load policy and env
        ray_config_path = os.path.join(expr_dir_path, 'params.pkl')
        ckpt_num = ckpt_dir_name.split("_")[-1]
        ckpt_filename = 'checkpoint-{}'.format(ckpt_num)
        ckpt_path = os.path.join(expr_dir_path, ckpt_dir_name, ckpt_filename)

        # load checkpoint
        with open(ray_config_path, 'rb') as ray_config_f:
            ray_config = pickle.load(ray_config_f)

        ray.init()

        # load policy and env
        env_config = ray_config['env_config']
        agent = ppo.PPOTrainer(config=ray_config, env=ray_config['env'])
        agent.restore(ckpt_path)
        env = ray_config['env'](env_config)

        # set seed and explore
        seed = args.seed if args.seed is not None else ray_config['seed']
        env.seed(seed)

        agent.get_policy().config['explore'] = args.explore

        # # run rollout episode + store logs
        run_rollouts(agent, env, DEFAULT_OUTPUT + "/eval.log")

        # store trajectory data

        # exit loop after desired number of trajectories collected
        ckpts_processed += 1
        if ckpts_processed == num_ckpts:
            break

    # create plot


if __name__ == "__main__":
    main()
