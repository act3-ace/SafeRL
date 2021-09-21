"""
This module is responsible for consuming experiment directories and creating trajectory data from evaluation episodes.
This data is then used to create trajectory plots. This is useful for showing policy improvements throughout training
and creating figures for our public paper.

Author: John McCarroll
"""

import math
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import jsonlines
import os
import numpy as np
import argparse
from glob import glob
import pickle5 as pickle
import ray
import ray.rllib.agents.ppo as ppo

from scripts.eval import run_rollouts, verify_experiment_dir
from saferl.environment.utils import YAMLParser, build_lookup, dict_merge


# Define Defaults
DEFAULT_NUM_CKPTS = 5
DEFAULT_SEED = 33
DEFAULT_OUTPUT = "/figures/data"
DEFAULT_TASK = "docking"
alt_env_config = None

# path to training logs
expr_dir = "/home/john/AFRL/Dubins/have-deepsky/scripts/output/docking_dev_data_1000iters_complete_ep"
DEFAULT_OUTPUT = "/figures/limit_data"
DEFAULT_CKPTS = [0, 1, 4, 14, 35]
alt_env_config = "/home/john/AFRL/Dubins/have-deepsky/configs/docking/docking_default.yaml"
only_plot = False
# only_plot = True


def get_args():
    """
    A function to process script args.

    Returns
    -------
    argparse.Namespace
        Collection of command line arguments and their values
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default="", help="The full path to the experiment directory") #, required=True)
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


def parse_log_trajectories(data_dir_path: str, environment_objs: list, iters: dict):
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
                x_dot = state["info"]["deputy"]["x_dot"]
                y_dot = state["info"]["deputy"]["y_dot"]

                trajectories[iter_num]["velocity"].append(math.sqrt(x_dot**2 + y_dot**2))
                trajectories[iter_num]["distance"].append(state["info"]["status"]["docking_distance"])
                trajectories[iter_num]["vel_limit"].append(state["info"]["status"]["max_vel_limit"])

    return trajectories


class Marker:
    def __init__(self, x1, y1, x2, y2, marker_type):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.marker_type = marker_type


# def parse_log_markers(trajectories: dict, env_objs: list):
#     # set up markers dict
#     markers = {}
#     for iter_num in trajectories.keys():
#         markers[iter_num] = {}
#         for obj in env_objs:
#             markers[iter_num][obj] = []
#
#     # collect points along trajectories
#     for iter_num in trajectories.keys():
#
#         for obj in env_objs:
#             # get traj
#             x_coords = trajectories[iter_num][obj]['x']
#             y_coords = trajectories[iter_num][obj]['y']
#
#             # add beginning and end points
#             markers[iter_num][obj].append(Marker(x_coords[0], y_coords[0], x_coords[1], y_coords[1], "start"))
#             markers[iter_num][obj].append(Marker(x_coords[-2], y_coords[-2], x_coords[-1], y_coords[-1], "end"))     # TODO: success / fail?
#
#             # add intermediate points
#             has_traj_ended = False
#             index = DEFAULT_MARKER_FREQ
#             while not has_traj_ended:
#                 # create Marker and add to list
#                 arrow = Marker(x_coords[index], y_coords[index], x_coords[index + 1], y_coords[index + 1], "arrow")
#                 markers[iter_num][obj].append(arrow)
#
#                 # update flag and index
#                 index += DEFAULT_MARKER_FREQ
#                 if index >= len(x_coords) - 2:
#                     has_traj_ended = True
#     return markers


def plot_data(data,
              output_filename=None,
              legend=None,
              task='docking',
              # agent=None,
              # target=None,
              # markers=None
              ):

    # set seaborn theme
    sns.set_theme()

    # create figure
    fig, ax = plt.subplots()

    # create color map + set scale
    cmap = plt.cm.get_cmap('plasma')  # cool, spring
    max_iter_num = 0
    for iter_num in data:
        if iter_num > max_iter_num:
            max_iter_num = iter_num

    # plot each eval
    line_num = 0
    for iter_num in sorted(list(data.keys())):
        # get color
        # color = cmap(iter_num / max_iter_num)
        line_num += 1
        color = cmap(line_num / 5)

        ax.plot(data[iter_num]['distance'], data[iter_num]['velocity'], color=color)

    # plot vel limit
    ax.plot(data[max_iter_num]['distance'], data[max_iter_num]['vel_limit'], color="black", linestyle='--')

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

    # axes scales
    # ax.set_aspect('equal', adjustable='box')

    # # task specific goal representation
    # # TODO: get radius from data
    # if task == 'docking':
    #     # add docking region
    #     docking_region = plt.Circle((0,0), 0.5, color='r')
    #     ax.add_patch(docking_region)
    # elif task == 'rejoin':
    #     pass

    # directionality markers
    # TODO: failure markers*

    # save figure
    if not output_filename:
        fig.savefig(output_filename)

    # show figure
    plt.show()


def get_iters(ckpt_num, expr_dir_path):
    progress_file = expr_dir_path + "/progress.csv"

    with open(progress_file, newline='') as csvfile:
        reader = csv.reader(csvfile)

        # define indices of interest
        agent_timesteps_total = 7
        training_iteration = 10
        ckpt_num = int(ckpt_num)

        for row in reader:
            if str(ckpt_num) == row[training_iteration]:
                return int(row[agent_timesteps_total])


def main():
    # collect experiment path
    args = get_args()
    # expr_dir_path = args.dir
    expr_dir_path = expr_dir

    # locate checkpoints
    expr_dir_path = verify_experiment_dir(expr_dir_path)
    ckpt_dirs = sorted(glob(expr_dir_path + "/checkpoint_*"))

    # create output dir
    output_path = expr_dir_path + args.output
    os.makedirs(output_path, exist_ok=True)

    iters = {}
    for ckpt_index in DEFAULT_CKPTS:
        # get filename of specified ckpts
        ckpt_dir_name = ckpt_dirs[ckpt_index]
        ray_config_path = os.path.join(expr_dir_path, 'params.pkl')
        ckpt_num = ckpt_dir_name.split("_")[-1].lstrip('0')  # remove trailing ckpt number from file and clean
        ckpt_filename = 'checkpoint-{}'.format(ckpt_num)
        ckpt_path = os.path.join(expr_dir_path, ckpt_dir_name, ckpt_filename)

        if not only_plot:
            # load checkpoint
            with open(ray_config_path, 'rb') as ray_config_f:
                ray_config = pickle.load(ray_config_f)

            ray.init(ignore_reinit_error=True)

            # load policy and env
            if alt_env_config:
                parser = YAMLParser(yaml_file=alt_env_config, lookup=build_lookup())
                config = parser.parse_env()
                env_config = config["env_config"]
            else:
                env_config = ray_config['env_config']

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
    environment_objs = ["deputy"]
    data = parse_log_trajectories(output_path, environment_objs, iters)

    ## plot data in matplotlib
    output_filename = output_path + "/figure1"
    plot_data(data, output_filename=output_filename, legend=iters)



    #
    # # create env obj list
    # environment_objs = None
    # if args.task == "docking":
    #     title = "Docking Trajectories Training Progress"
    #     environment_objs = ["chief", "deputy"]
    #     agent = "deputy"
    #     target = "chief"
    #
    # elif args.task == "rejoin":
    #     title = "Rejoin Trajectories Training Progress"
    #     environment_objs = ["lead", "wingman", "rejoin_region"]
    #     agent = "wingman"
    #     target = "lead"
    #

    #
    # # parse logs for trajectory data
    # trajectories = parse_log_trajectories(output_path, environment_objs, iters)
    #
    # # parse trajectories for extra info
    # markers = parse_log_markers(trajectories, environment_objs)
    #
    # # create plot
    # output_filename = output_path + "/../figure1"
    # plot_data(trajectories,
    #           output_filename=output_filename,
    #           # title=title,
    #           task=args.task,
    #           agent=agent,
    #           target=target,
    #           legend=iters,
    #           markers=markers
    #           )


if __name__ == "__main__":
    main()
