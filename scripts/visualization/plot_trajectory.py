"""
This module is responsible for consuming experiment directories and creating trajectory data from evaluation episodes.
This data is then used to create trajectory plots. This is useful for showing policy improvements throughout training
and creating figures for our public paper.

Author: John McCarroll
"""

import math
import csv
import seaborn as sns
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jsonlines
import os
import argparse
from glob import glob
import pickle5 as pickle
# import pickle
import ray
import ray.rllib.agents.ppo as ppo

from scripts.eval import run_rollouts, verify_experiment_dir
from saferl.environment.utils import YAMLParser, build_lookup


# Define Defaults
DEFAULT_SEED = 33
DEFAULT_OUTPUT = "/figures/data"
DEFAULT_TASK = "docking"
DEFAULT_MARKER_FREQ = 50
DEFAULT_CKPTS = [0, 1, 2, 3, 4]
DEFAULT_TRIAL_INDEX = 6


def get_args():
    """
    A function to process script args.

    Returns
    -------
    argparse.Namespace
        Collection of command line arguments and their values
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default="", help="The path to the experiment directory", required=True)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help="The seed used to initialize the evaluation environment")
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help="The location of logs from evaluation episodes")
    parser.add_argument('--task', type=str, default=DEFAULT_TASK,
                        help="The task on which the policy was trained and will be evaluated")
    parser.add_argument('--only_plot', action="store_true",
                        help="If evaluation data already generated and user needs quick plot generation.")
    parser.add_argument('--marker_freq', type=int, default=DEFAULT_MARKER_FREQ,
                        help="The frequency directional markers are drawn along trajectory lines.")
    parser.add_argument('--checkpoints', type=int, nargs="+", default=DEFAULT_CKPTS,
                        help="A list of checkpoint indices, from the experiment directory, to plot.")
    parser.add_argument('--trial_index', type=int, default=DEFAULT_TRIAL_INDEX,
                        help="The index corresponding to the desired experiment to load. "
                             "Use when multiple experiments are run by Tune.")
    parser.add_argument('--alt_env_config', type=str, default=None,
                        help="The path to an alternative config from which to run all trajectory episodes.")

    return parser.parse_args()


def parse_log_trajectories(data_dir_path: str, environment_objs: list, iters: dict):
    trajectories = {}

    # iterate through eval logs from data dir
    for ckpt_num, iter_num in iters.items():
        trajectories[iter_num] = {}

        # collect traj for each environment object of interest
        # but only one episode of target obj
        for obj in environment_objs:
            trajectories[iter_num][obj] = {
                'x': [],
                'y': []
            }

            # open eval log file
            with jsonlines.open(data_dir_path + "/eval{}.log".format(ckpt_num), 'r') as log:
                # iterate over json dict states
                for state in log:
                    x = state["info"][obj]["x"]
                    y = state["info"][obj]["y"]

                    # trajectories["vehicle"].append(vehicle)
                    trajectories[iter_num][obj]['x'].append(x)
                    trajectories[iter_num][obj]['y'].append(y)

    return trajectories


class Marker:
    def __init__(self, x1, y1, x2, y2, marker_type):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.marker_type = marker_type


def parse_log_markers(trajectories: dict, env_objs: list, marker_freq=100):
    # set up markers dict
    markers = {}
    for iter_num in trajectories.keys():
        markers[iter_num] = {}
        for obj in env_objs:
            markers[iter_num][obj] = []

    # collect points along trajectories
    for iter_num in trajectories.keys():

        for obj in env_objs:
            # get traj
            x_coords = trajectories[iter_num][obj]['x']
            y_coords = trajectories[iter_num][obj]['y']

            # add beginning and end points
            markers[iter_num][obj].append(Marker(x_coords[0], y_coords[0], x_coords[1], y_coords[1], "start"))
            markers[iter_num][obj].append(Marker(x_coords[-2], y_coords[-2], x_coords[-1], y_coords[-1], "end"))     # TODO: success / fail?

            # add intermediate points
            index = marker_freq
            while True:
                # check if index out of bounds
                if index >= len(x_coords) - 2:
                    break

                # create Marker and add to list
                arrow = Marker(x_coords[index], y_coords[index], x_coords[index + 1], y_coords[index + 1], "arrow")
                markers[iter_num][obj].append(arrow)

                # update index
                index += marker_freq

    return markers


def plot_data(data,
              output_filename=None,
              legend=None,
              task='docking',
              agent=None,
              target=None,
              markers=None):

    # set seaborn theme
    sns.set_theme()

    # create figure
    fig, ax = plt.subplots()

    # create color map + set scale
    cmap = plt.cm.get_cmap('plasma')        # cool, spring

    # plot each trajectory onto figure
    longest_episode = None
    longest_episode_len = 0
    line_num = 0
    for iter_num in sorted(list(data.keys())):
        # get color
        line_num += 1
        color = cmap(line_num / len(data))

        # plot each agent trajectory
        ax.plot(data[iter_num][agent]['x'], data[iter_num][agent]['y'], color=color)

        # plot arrow markers
        for marker in markers[iter_num][agent]:
            if marker.marker_type == "arrow":
                # calc angle
                x1 = marker.x1
                x2 = marker.x2
                y1 = marker.y1
                y2 = marker.y2

                start_angle = (math.atan2(y2-y1, x2-x1) * 180 / math.pi) - 90

                # place marker on plot
                ax.scatter(x=x1, y=y1, marker=(3, 0, start_angle), color=color)

        # record longest episode
        if len(data[iter_num][target]['x']) > longest_episode_len:
            longest_episode = iter_num
            longest_episode_len = len(data[iter_num][target]['x'])

    # plot example target
    ax.plot(data[longest_episode][target]['x'], data[longest_episode][target]['y'], color='black', linestyle='--')

    if task == "rejoin":
        # TODO: abstract duplicate code
        for marker in markers[longest_episode][target]:
            if marker.marker_type == "arrow":
                # calc angle
                x1 = marker.x1
                x2 = marker.x2
                y1 = marker.y1
                y2 = marker.y2

                start_angle = (math.atan2(y2 - y1, x2 - x1) * 180 / math.pi) - 90

                # place marker on plot
                ax.scatter(x=x1, y=y1, marker=(3, 0, start_angle), color='black')

    # titles
    axes_font_dict = {
        'fontstyle': 'italic',
        'fontsize': 10
    }
    # title_font_dict = {
    #     'fontweight': 'bold',
    #     'fontsize': 10
    # }

    plt.xlabel("X", fontdict=axes_font_dict)
    plt.ylabel("Y", fontdict=axes_font_dict)
    # plt.title(title, fontdict=title_font_dict)

    # legend
    legend_list = [iter_num for iter_num in sorted(list(legend.values()))]
    for i, number in enumerate(legend_list):
        if legend_list[i] < 1000000 and legend_list[i] >= 1000:
            legend_list[i] = str(int(legend_list[i] / 1000)) + 'k'
        elif legend_list[i] > 1000000:
            legend_list[i] = str(legend_list[i] / 1000000)[0:4] + 'M'

    if task == "rejoin":
        legend_list.append('lead')

    ax.legend(legend_list)

    plt.tight_layout(pad=0.5)

    # axes scales
    ax.set_aspect('equal', adjustable='box')

    # task specific goal representation
    # TODO: get radius from data
    if task == 'docking':
        # add docking region
        docking_region = plt.Circle((0, 0), 0.5, color='r')
        ax.add_patch(docking_region)
    elif task == 'rejoin':
        pass

    # TODO: failure markers*

    # save figure
    if output_filename:
        fig.savefig(output_filename)

    # show figure
    plt.show()


def get_iters(ckpt_num, expr_dir_path):
    progress_file = expr_dir_path + "/progress.csv"

    with open(progress_file, newline='') as csvfile:
        reader = csv.reader(csvfile)

        # define indices of interest
        column_headers = next(iter(reader))
        timesteps_total_index = column_headers.index("timesteps_total")
        training_iteration_index = column_headers.index("training_iteration")
        ckpt_num = int(ckpt_num)

        for row in reader:
            if str(ckpt_num) == row[training_iteration_index]:
                return int(row[timesteps_total_index])


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

    # create env obj list
    environment_objs = None
    if args.task == "docking":
        title = "Docking Trajectories Training Progress"
        environment_objs = ["chief", "deputy"]
        agent = "deputy"
        target = "chief"

    elif args.task == "rejoin":
        title = "Rejoin Trajectories Training Progress"
        environment_objs = ["lead", "wingman", "rejoin_region"]
        agent = "wingman"
        target = "lead"

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

            # HACKY DOCKING FIX
            del ray_config["callbacks"]
            # del ray_config["render_env"]
            # del ray_config["record_env"]
            # del ray_config["placement_strategy"]

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

    # parse logs for trajectory data
    trajectories = parse_log_trajectories(output_path, environment_objs, iters)

    # parse trajectories for extra info
    markers = parse_log_markers(trajectories, environment_objs, args.marker_freq)

    # create plot
    #output_filename = output_path + "/../figure1"
    output_filename = "./figs/figure1.png"
    plot_data(trajectories,
              output_filename=output_filename,
              # title=title,
              task=args.task,
              agent=agent,
              target=target,
              legend=iters,
              markers=markers
              )


if __name__ == "__main__":
    main()
