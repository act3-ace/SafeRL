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
    parser.add_argument('--only_plot', action="store_true",
                        help="If evaluation data already generated and user needs quick plot generation.")
    # TODO: 2d vs 3d flag?

    return parser.parse_args()


def parse_log_trajectories(data_dir_path: str, num_ckpts: int, environment_objs: list, task='docking'):
    trajectories = {
        "vehicle": [],
        "x": [],
        "y": []
    }

    # track first episode
    is_first_episode_done = False

    # iterate through eval logs from data dir
    for i in range(1, num_ckpts + 1):

        # open eval log file
        with jsonlines.open(data_dir_path + "/eval{}.log".format(i), 'r') as log:       # TODO: get correct ckpts

            # iterate over json dict states
            for state in log:
                # collect traj for each environment object of interest
                # but only one episode of target obj
                for obj in environment_objs:
                    if is_first_episode_done and (obj == 'lead' or obj == 'chief'):
                        # skip data collection if already got lead / chief data
                        continue

                    x = state["info"][obj]["x"]
                    y = state["info"][obj]["y"]

                    # store coords
                    trajectories["vehicle"].append(obj + str(i))
                    trajectories["x"].append(x)
                    trajectories["y"].append(y)

                # track first episode
                if state["info"]["success"] is True or state["info"]["failure"] is not False:
                    is_first_episode_done = True

    return trajectories


# Nate's plotting method
def plot_data(data,
              xaxis='x',
              value='y',
              condition="vehicle",
              output_filename=None,
              title="Trajectories",
              task='docking',
              xmax=None,
              ylim=None,
              **kwargs):

    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(data=data, x=xaxis, y=value, hue=condition, sort=False, markers="^", style=[])    # ci='sd'
    plt.legend(loc='best', prop={'size': 8}).set_draggable(True)

    # if xmax is None:
    #     xmax = np.max(np.asarray(data[xaxis]))
    # plt.xlim(right=xmax)
    #
    # if ylim is not None:
    #     plt.ylim(ylim)

    # xscale = xmax > 5e3
    # if xscale:
    #     # Just some formatting niceness: x-axis scale in scientific notation if max x is large
    #     plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # titles
    axes_font_dict = {
        'fontstyle': 'italic',
        'fontsize': 10
    }
    title_font_dict = {
        'fontweight': 'bold',
        'fontsize': 15
    }

    plt.xlabel("X", fontdict=axes_font_dict)
    plt.ylabel("Y", fontdict=axes_font_dict)
    plt.title(title, fontdict=title_font_dict)

    plt.tight_layout(pad=0.5)

    # axes scales
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # task specific goal representation
    if task == 'docking':
        pass
    elif task == 'rejoin':
        pass

    # save figure
    if not output_filename:
        fig = plt.gcf()
        fig.savefig(output_filename)

    # show figure
    plt.show()


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
        title = "Docking Trajectories Training Progress"
        environment_objs = ["chief", "deputy"]

    elif args.task == "rejoin":
        title = "Rejoin Trajectories Training Progress"
        environment_objs = ["lead", "wingman"]

    ckpts_processed = 0
    if not args.only_plot:
        for ckpt_dir_name in ckpt_dirs:
            # iterate through checkpoints
            # assuming ckpt_dirs ordered from latest to earliest

            # load policy and env
            ray_config_path = os.path.join(expr_dir_path, 'params.pkl')
            ckpt_num = ckpt_dir_name.split("_")[-1].lstrip('0')           # remove trailing ckpt number from file and clean
            ckpt_filename = 'checkpoint-{}'.format(ckpt_num)
            ckpt_path = os.path.join(expr_dir_path, ckpt_dir_name, ckpt_filename)

            # load checkpoint
            with open(ray_config_path, 'rb') as ray_config_f:
                ray_config = pickle.load(ray_config_f)

            ray.init(ignore_reinit_error=True)

            # load policy and env
            env_config = ray_config['env_config']
            agent = ppo.PPOTrainer(config=ray_config, env=ray_config['env'])
            agent.restore(ckpt_path)
            env = ray_config['env'](env_config)

            # set seed
            seed = args.seed if args.seed is not None else ray_config['seed']
            env.seed(seed)

            # delete any previous logs
            log_dir = output_path + "/eval{}.log".format(num_ckpts - ckpts_processed)
            if os.path.isfile(log_dir):
                os.remove(log_dir)

            # run rollout episode + store logs
            run_rollouts(agent, env, log_dir)

            # exit loop after desired number of trajectories collected
            ckpts_processed += 1
            if ckpts_processed == num_ckpts:
                break

    # parse logs for trajectory data
    trajectories = parse_log_trajectories(output_path, num_ckpts, environment_objs)

    # create plot
    output_filename = output_path + "/../figure1"
    plot_data(trajectories,
              xaxis="x",
              value="y",
              condition="vehicle",
              output_filename=output_filename,
              title=title,
              task=args.task)


if __name__ == "__main__":
    main()


# TODO:
# could have target + agent keys in env obj dict to flexibly determine which vehicle traj should only be recorded once?
# could hardcode into parse_log logic?

# ultimately we need differences in rendering docking vs. rejoin plots
# plotting rejoin region vs docking region
# and plotting target / target trajectories
# maybe these should be handled by same encapsulated func (mini strat pattern?)

# then color / direction are universal
