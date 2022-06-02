import math
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
import argparse
from tqdm import tqdm

from scripts.eval import parse_jsonlines_log


def get_args():
    """
    A function to process script args.

    Returns
    -------
    argparse.Namespace
        Collection of command line arguments and their values
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('log', type=str, help="The full path to the trajectory log file")
    parser.add_argument('plot_subject', type=str, help="name of object to plot")

    return parser.parse_args()


def plot_trajectories(episode_logs, subject, mode="2d"):
    fig, ax = plt.subplots()
    ax.set_aspect(aspect='equal', adjustable="datalim")

    for log in episode_logs:
        positions = [[step_log['info'][subject]["x"], step_log['info'][subject]["y"], step_log['info'][subject]["z"]]
                     for step_log in log]

        trajectory = np.array(positions, dtype=float)

        if mode == "2d":
            ax.plot(trajectory[:, 0], trajectory[:, 1])
        else:
            raise ValueError(f"mode {mode} not supported. Must be ['2d']")

    # plt.show()
    plt.savefig("trajectories.png", dpi=200)


def main():
    # process args
    args = get_args()
    episode_logs = parse_jsonlines_log(args.log, separate_episodes=True)

    plot_trajectories(episode_logs, subject=args.plot_subject)


if __name__ == "__main__":
    main()
