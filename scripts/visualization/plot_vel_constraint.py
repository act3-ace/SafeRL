"""
This module is responsible for consuming experiment directories and creating velocity data from evaluation episodes.
This data is then used to create velocity plots vs the velocity constraint. This is useful for showing policy
improvements throughout training and creating figures for our public paper.

Author: John McCarroll
"""

import math
import seaborn as sns
import matplotlib.pyplot as plt
import jsonlines
import os
import argparse
from glob import glob
import pickle5 as pickle
import ray
import ray.rllib.agents.ppo as ppo

from scripts.eval import run_rollouts, verify_experiment_dir
from scripts.visualization.plot_trajectory import get_iters
from saferl.environment.utils import YAMLParser, build_lookup


# Define Defaults
DEFAULT_SEED = 0
DEFAULT_OUTPUT = "/figures/limit_data"
DEFAULT_TASK = "docking"
DEFAULT_CKPTS = [0, 1, 2, 3, 4]
DEFAULT_TRIAL_INDEX = 0


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
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help="The seed used to initialize the evaluation environment")
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help="The location of logs from evaluation episodes")
    parser.add_argument('--only_plot', action="store_true",
                        help="If evaluation data already generated and user needs quick plot generation.")
    parser.add_argument('--checkpoints', type=int, nargs="+", default=DEFAULT_CKPTS,
                        help="A list of checkpoint indices, from the experiment directory, to plot.")
    parser.add_argument('--alt_env_config', type=str, default=None,
                        help="The path to an alternative config from which to run all trajectory episodes.")
    parser.add_argument('--trial_index', type=int, default=DEFAULT_TRIAL_INDEX,
                        help="The index corresponding to the desired experiment to load. "
                             "Use when multiple trials are run by Tune.")

    return parser.parse_args()


def parse_log_trajectories(data_dir_path: str, agent_name: str, iters: dict):
    trajectories = {}

    # iterate through eval logs from data dir
    for ckpt_num, iter_num in iters.items():
        trajectories[iter_num] = {
            "velocity": [],
            "distance": [],
            "vel_limit": []
        }

        # open eval log file
        with jsonlines.open(data_dir_path + "/eval{}.log".format(ckpt_num), 'r') as log:
            # iterate over json dict states
            for state in log:
                x_dot = state["info"][agent_name]["x_dot"]
                y_dot = state["info"][agent_name]["y_dot"]

                trajectories[iter_num]["velocity"].append(math.sqrt(x_dot**2 + y_dot**2))
                trajectories[iter_num]["distance"].append(state["info"]["status"]["docking_distance"])
                trajectories[iter_num]["vel_limit"].append(state["info"]["status"]["max_vel_limit"])

    return trajectories


def plot_data(data,
              output_filename=None,
              legend=None
              ):

    # set seaborn theme
    sns.set_theme()

    # create figure
    fig, ax = plt.subplots()

    # create color map + set scale
    cmap = plt.cm.get_cmap('rainbow')  # cool, spring, plasma

    # plot each eval
    line_num = 0
    largest_distance_span = 0
    largest_distance_span_episode = None
    for iter_num in sorted(list(data.keys())):
        # get color
        line_num += 1
        color = cmap(line_num / len(data))

        ax.plot(data[iter_num]['distance'], data[iter_num]['velocity'], color=color)

        # record largest distance span episode
        distance_span = max(data[iter_num]['distance']) - min(data[iter_num]['distance'])
        if distance_span > largest_distance_span:
            largest_distance_span_episode = iter_num
            largest_distance_span = distance_span

    # plot vel limit
    ax.plot(data[largest_distance_span_episode]['distance'], data[largest_distance_span_episode]['vel_limit'], color="black", linestyle='--')

    axes_font_dict = {
        'fontstyle': 'italic',
        'fontsize': 10
    }
    # title_font_dict = {
    #     'fontweight': 'bold',
    #     'fontsize': 10
    # }

    plt.xlabel("Distance from Chief (m)", fontdict=axes_font_dict)
    plt.ylabel("Velocity (m/s)", fontdict=axes_font_dict)
    # plt.title(title, fontdict=title_font_dict)

    # legend
    legend_list = [iter_num for iter_num in sorted(list(legend.values()))]
    for i, number in enumerate(legend_list):
        if legend_list[i] < 1000000 and legend_list[i] >= 1000:
            legend_list[i] = str(int(legend_list[i] / 1000)) + 'k'
        elif legend_list[i] > 1000000:
            legend_list[i] = str(legend_list[i] / 1000000)[0:4] + 'M'

    ax.legend(legend_list)

    plt.tight_layout(pad=0.5)

    # save figure
    if output_filename:
        fig.savefig(output_filename)

    # show figure
    plt.show()


def main():
    # collect experiment path
    args = get_args()
    expr_dir_path = args.dir

    # locate checkpoints
    expr_dir_path = verify_experiment_dir(expr_dir_path, trial_index=args.trial_index)
    ckpt_dirs = sorted(glob(expr_dir_path + "/checkpoint_*"),
                       key=lambda ckpt_dir_name: int(ckpt_dir_name.split("_")[-1]))

    # create output dir
    output_path = expr_dir_path + args.output
    os.makedirs(output_path, exist_ok=True)

    iters = {}
    for ckpt_index in args.checkpoints:
        # get filename of specified ckpts
        ckpt_dir_name = ckpt_dirs[ckpt_index]
        ray_config_path = os.path.join(expr_dir_path, 'params.pkl')
        ckpt_num = ckpt_dir_name.split("_")[-1].lstrip('0')  # remove trailing ckpt number from file and clean
        ckpt_filename = 'checkpoint-{}'.format(ckpt_num)
        ckpt_path = os.path.join(ckpt_dir_name, ckpt_filename)

        if not args.only_plot:
            # load checkpoint
            with open(ray_config_path, 'rb') as ray_config_f:
                ray_config = pickle.load(ray_config_f)

            ray.init(ignore_reinit_error=True)

            # load policy and env
            if args.alt_env_config:
                parser = YAMLParser(yaml_file=args.alt_env_config, lookup=build_lookup())
                config = parser.parse_env()
                env_config = config["env_config"]
            else:
                env_config = ray_config['env_config']

            # HACKY FIX
            del ray_config["callbacks"]

            rl_agent = ppo.PPOTrainer(config=ray_config, env=ray_config['env'])
            rl_agent.restore(ckpt_path)
            env = ray_config['env'](env_config)

            # set seed
            seed = args.seed if args.seed is not None else ray_config['seed']
            env.seed(seed)

            # delete any previous logs
            log_dir = output_path + "/eval{}.log".format(ckpt_num)
            if os.path.isfile(log_dir):
                os.remove(log_dir)

            # run rollout episode + store logs
            run_rollouts(rl_agent, env, log_dir)

        # collect number of env iters coresponding to ckpt num
        iters[ckpt_num] = get_iters(ckpt_num, expr_dir_path)

    # parse eval logs for vel data
    agent_name = "deputy"
    data = parse_log_trajectories(output_path, agent_name, iters)

    ## plot data in matplotlib
    # output_filename = output_path + "/figure1"
    os.makedirs('./figs', exist_ok=True)
    output_filename = "./figs/vel_constr.png"
    plot_data(data, output_filename=output_filename, legend=iters)


if __name__ == "__main__":
    main()
