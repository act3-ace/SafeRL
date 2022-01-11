"""

Author: John McCarroll
10-14-2021
"""

import pandas as pd
import numpy as np
import jsonlines
from flatten_json import flatten_json
import time
import pickle
import matplotlib.pyplot as pyplot


def process_log(path_to_file: str, blacklist: list, is_jsonlines=True):
    """
    This function handles the conversion of stored historical trial data to in-memory python objects,
    namely a pandas.DataFrame metadata table and a dictionary of episode ID to episode summary DataFrames.
    """

    # index (row #) -> dict (flattened state dict) [step_number | episode_ID | wingman_x | ... ]
    data = {
        "wingman": {
            "x": [],
            "y": [],
            "z": [],
            "heading": [],
            "gamma": [],
            "roll": []
        },
        "rejoin_region": {
            "x": [],
            "y": [],
            "z": []
        },
        "time": []
    }

    t_start = time.time()
    # open log file
    if is_jsonlines:
        with jsonlines.open(path_to_file, 'r') as log:
            # iterate through json objects in log
            for state in log:
                data["wingman"]["x"].append(state["info"]["wingman"]["x"])
                data["wingman"]["y"].append(state["info"]["wingman"]["y"])
                data["wingman"]["z"].append(state["info"]["wingman"]["z"])
                data["wingman"]["heading"].append(state["info"]["wingman"]["heading"])
                data["wingman"]["gamma"].append(state["info"]["wingman"]["gamma"])
                data["wingman"]["roll"].append(state["info"]["wingman"]["roll"])

                data["rejoin_region"]["x"].append(state["info"]["rejoin_region"]["x"])
                data["rejoin_region"]["y"].append(state["info"]["rejoin_region"]["y"])
                data["rejoin_region"]["z"].append(state["info"]["rejoin_region"]["z"])

                data["time"].append(state["step_number"] * state["info"]["timestep_size"])

    return data


def plot(data):
    # plot x,y,z
    fig, ax = pyplot.subplots()

    time = data["time"]
    x = data["wingman"]["x"]

    ax.plot(time, x, label="x")

    ax.set_xlabel("time")
    ax.set_title("Episode {}:\n Wingman X vs. Time")

    fig.savefig("./x_plot.png")

    # plot y
    fig, ax = pyplot.subplots()

    y = data["wingman"]["y"]

    ax.plot(time, y, label="y")

    ax.set_xlabel("time")
    ax.set_title("Wingman Y vs. Time")

    fig.savefig("./y_plot.png")

    # plot z
    fig, ax = pyplot.subplots()

    z = data["wingman"]["z"]

    ax.plot(time, z, label="z")

    ax.set_xlabel("time")
    ax.set_title("Wingman Z vs. Time")

    fig.savefig("./z_plot.png")

    # plot heading
    fig, ax = pyplot.subplots()

    heading = data["wingman"]["heading"]

    ax.plot(time, heading, label="heading")

    ax.set_xlabel("time")
    ax.set_title("Wingman heading vs. Time")

    fig.savefig("./heading_plot.png")

    # plot gamma
    fig, ax = pyplot.subplots()

    gamma = data["wingman"]["gamma"]

    ax.plot(time, gamma, label="gamma")

    ax.set_xlabel("time")
    ax.set_title("Wingman gamma vs. Time")

    fig.savefig("./gamma_plot.png")

    # plot roll
    fig, ax = pyplot.subplots()

    roll = data["wingman"]["roll"]

    ax.plot(time, roll, label="roll")

    ax.set_xlabel("time")
    ax.set_title("Wingman roll vs. Time")

    fig.savefig("./roll_plot.png")

    



def main():
    # dir = "/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/docking_3d/expr_20211014_181535/PPO_DockingEnv_bb20d_00000_0_seed=913840577_2021-10-14_18-15-37/"
    dir = "/data/petabyte/safe_autonomy/ieee_aero_2022/experiments/rejoin_3d_tall/expr_20211015_043032/PPO_DubinsRejoin_a3c6e_00000_0_seed=913840577_2021-10-15_04-30-35/figures/data/eval500.log"

    data = process_log(dir, [])
    plot(data)


if __name__ == "__main__":
    main()
