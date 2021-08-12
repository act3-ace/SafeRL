"""
This script implements the API used by the plot_notebook.ipynb Jupyter notebook in order to provide quick and powerful
insight into experiment log data. In the future, running this script directly will launch a simplified plotting tool,
however currently such a feature is not implemented.

Note: requires flatten_json package amd console-menu
        (pip install flatten_json
         pip install console-menu)

Author: John McCarroll
3-5-2021
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
    metadata_table = {
        "worker_episode_number": [],
        "episode_ID": [],
        "episode_duration": [],
        "episode_success": [],
        "episode_failure": []
    }

    # index (row #) -> dict (flattened state dict) [step_number | episode_ID | wingman_x | ... ]
    episode_dictionaries = {}

    t_start = time.time()
    # open log file
    if is_jsonlines:
        with jsonlines.open(path_to_file, 'r') as log:
            episode_duration = 0
            prev_ID = None
            prev_success = None
            prev_failure = None
            prev_episode_number = None

            # iterate through json objects in log
            for state in log:
                episode_success = state["info"]["success"]
                episode_failure = state["info"]["failure"]
                episode_ID = state["episode_ID"]

                # apply blacklist filter
                for unwanted_key in blacklist:
                    state.pop(unwanted_key, None)

                # start of new episode
                if episode_ID not in episode_dictionaries:

                    # store previous episode's metadata
                    if prev_ID:
                        metadata_table["worker_episode_number"].append(prev_episode_number)
                        metadata_table["episode_ID"].append(prev_ID)
                        metadata_table["episode_duration"].append(episode_duration)
                        metadata_table["episode_success"].append(prev_success)
                        metadata_table["episode_failure"].append(prev_failure)

                    # reset metadata counters
                    episode_duration = 0

                    # construct pandas table for new episode & place in map
                    episode_dictionaries[episode_ID] = {episode_duration: flatten_json(state)}

                    # continuing current episode
                else:
                    # update metadata counters
                    episode_duration += 1 * state["info"]["timestep_size"]

                    # add state to table
                    episode_dictionaries[episode_ID][episode_duration] = flatten_json(state)

                prev_ID = episode_ID
                prev_success = episode_success
                prev_failure = episode_failure
                prev_episode_number = state["worker_episode_number"]

        # construct episode DataFrames
        episode_dataframes = {}
        for ID, episode_dict in episode_dictionaries.items():
            episode_dataframes[ID] = pd.DataFrame.from_dict(episode_dict, "index")

        # construct metadata DataFrame
        metadata_df = pd.DataFrame(metadata_table)
        t_end = time.time()
        print("log read time.yaml: " + str(t_end - t_start))
    else:
        with open(path_to_file, "rb") as file:         # this has issues with saves using "log" in beginning
            episode_dataframes = pickle.load(file)

        with open(path_to_file.strip("log")+"meta", "rb") as file:
            metadata_df = pickle.load(file)

    return metadata_df, episode_dataframes


def plot(x, y=None, ax=None):
    """
    This function is responsible for plotting data to a provided Axes object. If no Axes is provided, one will
    be created. The resulting Figure object is returned.
    Params:
    x  - a pandas Series, numpy array, or ndarray of numeric values
    y  - a dictionary of {string_name: Series} KVPs
    ax - an Axes object on which to plot the data
    """
    # make subplot
    if ax is None:
        fig, ax = pyplot.subplots()
    # plot on given Axes
    else:
        fig = ax.figure

    # check vars for dimensionality / multi-line
    if list == type(x):
        return fig

    if not y:
        # 1D plot of x array
        ax.plot(x)
    else:
        # 2D plot of lines within y
        if type(y) is dict:
            for label, data in y.items():
                ax.plot(x, data, label=label)

    return fig


def plot_variables(plot_name: str):
    """
    function to quickly plot variables on single plot
    """
    # gather data and col names
    global x_vars, y_vars, episode_dataframes, selected_episode, main_axes
    episode = episode_dataframes[selected_episode]

    for x in x_vars:
        for y in y_vars:
            x_array = episode[x].to_numpy()
            y_array = episode[y].to_numpy()
            plot(x_array, {y: y_array}, ax=main_axes)

    main_axes.figure.savefig(plot_name)


# TODO
def to_numpy(data):
    """
    Helper func to convert array-like data to plottable type numpy.ndarray
    """

    if type(data) is list:
        return np.ndarray(data)

    if type(data) is pd.DataFrame or type(data) is pd.Series:
        return data.to_numpy()
