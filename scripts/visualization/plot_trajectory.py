"""
This module is responsible for consuming experiment directories and creating trajectory data from evaluation episodes.
This data is then used to create trajectory plots. This is useful for showing policy improvements throughout training
and creating figures for our public paper.

Author: John McCarroll
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import jsonlines
import os
import os.path as osp
import numpy as np
import argparse
from glob import glob
import pickle5 as pickle
import ray
import ray.rllib.agents.ppo as ppo

from scripts.eval import run_rollouts, verify_experiment_dir


# Define Defaults
DEFAULT_NUM_CKPTS = 5
DEFAULT_SEED = 33
DEFAULT_OUTPUT = "/figures/data"
DEFAULT_TASK = "docking"


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
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help="The location of logs from evaluation episodes")
    parser.add_argument('--task', type=str, default=DEFAULT_TASK,
                        help="The task on which the policy was trained and will be evaluated")
    # TODO: 2d vs 3d flag?

    return parser.parse_args()


def parse_log_trajectories(data_dir_path: str, num_ckpts: int, environment_objs: list):
    trajectories = {
        "vehicle_ckpt": [],
        "x": [],
        "y": []
    }

    # iterate through eval logs from data dir
    for i in range(1, num_ckpts + 1):
        # # populate trajectories map with empty lists
        # for obj in environment_objs:
        #     trajectories[obj + str(i)] = {
        #         "x": [],
        #         "y": []
        #     }

        # open eval log file
        with jsonlines.open(data_dir_path + "/eval{}.log".format(i), 'r') as log:       # TODO: get correct ckpts
            # iterate over json dict states
            for state in log:
                # collect traj for each environment object of interest
                for obj in environment_objs:
                    x = state["info"][obj]["x"]
                    y = state["info"][obj]["y"]

                    # store coords
                    trajectories["vehicle_ckpt"].append(obj + str(i))
                    trajectories["x"].append(x)
                    trajectories["y"].append(y)

    return trajectories


# Nate's plotting method
def plot_data(data,
              xaxis='Epoch',
              value="AverageEpRet",         #TODO: change x and y defaults
              condition="Condition1",
              smooth=1,
              xmax=None,
              ylim=None,
              **kwargs):

    # if smooth > 1:
    #     """
    #     smooth data with moving window average.
    #     that is,
    #         smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
    #     where the "smooth" param is width of that window (2k+1)
    #     """
    #     y = np.ones(smooth)
    #     for datum in data:
    #         x = np.asarray(datum[value])
    #         z = np.ones(len(x))
    #         smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
    #         datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.5)
    # sns.tsplot(data=data, time=xaxis, value=value, unit="Unit", condition=condition, ci='sd', **kwargs)
    sns.relplot(data=data, x=xaxis, y=value, hue=condition, kind="line") #ci='sd', **kwargs)
    """
    If you upgrade to any version of Seaborn greater than 0.8.1, switch from 
    tsplot to lineplot replacing L29 with:

        sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci='sd', **kwargs)

    Changes the colorscheme and the default legend style, though.
    """
    plt.legend(loc='best').set_draggable(True)
    # plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    """
    For the version of the legend used in the Spinning Up benchmarking page, 
    swap L38 with:

    plt.legend(loc='upper center', ncol=6, handlelength=1,
               mode="expand", borderaxespad=0., prop={'size': 13})
    """
    if xaxis is 'TotalEnvInteracts':
        plt.xlabel('Timesteps')
    if value is 'AverageTestEpRet' or value is 'AverageAltTestEpRet':
        plt.ylabel('Average Return')
    if value is 'TestEpLen' or value is 'AltTestEpLen':
        plt.ylabel('Average Episode Length')

    if xmax is None:
        xmax = np.max(np.asarray(data[xaxis]))
    plt.xlim(right=xmax)

    if ylim is not None:
        plt.ylim(ylim)

    xscale = xmax > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    plt.tight_layout(pad=0.5)


def main():
    # collect experiment path
    args = get_args()
    expr_dir_path = args.dir
    num_ckpts = args.num_ckpts

    # locate checkpoints
    expr_dir_path = verify_experiment_dir(expr_dir_path)
    ckpt_dirs = glob(expr_dir_path + "/checkpoint_*")

    # create output dir
    output_path = expr_dir_path + args.output
    os.makedirs(output_path, exist_ok=True)

    # create env obj list
    # TODO: remove hardcoding (cmd line opt and/or read from env)
    environment_objs = None
    if args.task == "docking":
        environment_objs = ["chief", "deputy"]
    elif args.task == "rejoin":
        environment_objs = ["lead", "wingman"]

    ckpts_processed = 0
    # for ckpt_dir_name in ckpt_dirs:
    #     # iterate through checkpoints
    #     # assuming ckpt_dirs ordered from latest to earliest
    #
    #     # load policy and env
    #     ray_config_path = os.path.join(expr_dir_path, 'params.pkl')
    #     ckpt_num = ckpt_dir_name.split("_")[-1].lstrip('0')           # remove trailing ckpt number from file and clean
    #     ckpt_filename = 'checkpoint-{}'.format(ckpt_num)
    #     ckpt_path = os.path.join(expr_dir_path, ckpt_dir_name, ckpt_filename)
    #
    #     # load checkpoint
    #     with open(ray_config_path, 'rb') as ray_config_f:
    #         ray_config = pickle.load(ray_config_f)
    #
    #     ray.init(ignore_reinit_error=True)
    #
    #     # load policy and env
    #     env_config = ray_config['env_config']
    #     agent = ppo.PPOTrainer(config=ray_config, env=ray_config['env'])
    #     agent.restore(ckpt_path)
    #     env = ray_config['env'](env_config)
    #
    #     # set seed
    #     seed = args.seed if args.seed is not None else ray_config['seed']
    #     env.seed(seed)
    #
    #     # run rollout episode + store logs
    #     run_rollouts(agent, env, output_path + "/eval{}.log".format(num_ckpts - ckpts_processed))
    #
    #     # exit loop after desired number of trajectories collected
    #     ckpts_processed += 1
    #     if ckpts_processed == num_ckpts:
    #         break

    # parse logs for trajectory data
    trajectories = parse_log_trajectories(output_path, num_ckpts, environment_objs)

    # create plot
    # for i in range(1, num_ckpts + 1):
    plt.figure()
    plot_data(trajectories, xaxis="x", value="y", condition="vehicle_ckpt")

    plt.show()


if __name__ == "__main__":
    main()


# TODO:
#   get data in correct pandas dataframe format
#   create seaborn plot from data
#   make pretty
