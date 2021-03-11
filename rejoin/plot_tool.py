"""
Script to prototype a dev debugging tool that presents RLLib logs in a readable, informant and directed manner.

Note: requires flatten_json package
        (pip install flatten_json)
        + console-menu
        + ...

Author: John McCarroll
3-5-2021
"""

import pandas as pd
import numpy as np
import jsonlines
from matplotlib import pyplot
from flatten_json import flatten_json
import time
import h5py
import pickle
from consolemenu import *
from consolemenu.items import *


### Consume log file and const pandas tables TODO: relative paths
path_to_log = "/home/john/AFRL/Dubins/have-deepsky/rejoin/output/expr_20210308_085452/training_logs/worker_1.log"
# path_to_log = "/home/john/AFRL/Dubins/have-deepsky/rejoin/output/expr_20210308_112211/training_logs/worker_1.log"
# path_to_save = "/media/john/HDD/logging_test.hdf5"
blacklist = ["obs", "time"]              # list of log keys to omit from pandas table
load = True


metadata_table = {
    "worker_episode_number": [],
    "episode_ID": [],
    "episode_duration": [],
    "episode_success": [],
    "episode_failure": []

}
episode_dictionaries = {}                # index (row #) -> dict (flattened state dict) [step_number | episode_ID | wingman_x | ... ]

t_start = time.time()
# open log file
if not load:
    with jsonlines.open(path_to_log, 'r') as log:
        episode_duration = 0
        prev_ID = None
        prev_success = None
        prev_failure = None

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

    # construct meta data FataFrame
    metadata_df = pd.DataFrame(metadata_table)
    t_end = time.time()
    print("log read time: " + str(t_end - t_start))
else:
    with open("/media/john/HDD/Dubins_2D_preprocessed.log", "rb") as file:
        episode_dataframes = pickle.load(file)

    with open("/media/john/HDD/Dubins_2D_preprocessed.meta", "rb") as file:
        metadata_df = pickle.load(file)


pd.set_option("display.max_rows", None, "display.max_columns", 500, "display.width", 500)
# print(metadata_df)


### Create UI menu
## define global vars and functions
selected_episode = next(iter(episode_dataframes.keys()))
available_variables = next(iter(episode_dataframes.values())).columns.values
x_vars = {"step_number"}
y_vars = set()
z_vars = set()
var_map = {
    "x": x_vars,
    "y": y_vars,
    "z": z_vars,
}


def display_metadata():
    print(metadata_df)


def display_selected_episode():
    print(selected_episode)


def set_selected_episode():
    global selected_episode
    print("Enter Episode ID:")
    ep_ID_input = input()
    try:
        ep_ID_input = int(ep_ID_input.strip())
    except ValueError:
        print("Invalid ID format - must be integer")
        return

    if ep_ID_input in episode_dataframes:
        selected_episode = ep_ID_input
        print("Selected Episode {}".format(selected_episode))
    else:
        print("Entered ID does not match any in current log")


def manipulate_variables():
    print("Available Variables:\n")
    print(available_variables)
    print()
    print("Syntax: [x | y | z] [add | rm] [variable_name | all]")
    print("Enter 'done' to exit")
    print()

    global var_map, x_vars, y_vars, z_vars

    while True:
        print("Current Variables:\n x - {}\n y - {}\n z - {}".format(x_vars, y_vars, z_vars))
        print("Enter Command:\n")

        # take user input
        command = input().strip()
        # exit if desired
        if command == "done" or command == "exit" or command == "quit":
            break

        command = command.split(" ")
        # ensure command format
        if len(command) != 3:
            print("Expected 3 positional arguments, only found {}".format(len(command)))
            continue

        axis = command[0].strip()
        operation = command[1].strip()
        variable = command[2].strip()
        # ensure command sensible
        if axis != "x" and axis != "y" and axis != "z":
            print("Invalid axis: expected 'x', 'y', or 'z'")
            continue
        if variable not in available_variables and variable != "all":
            print("Unexpected variable name")
            continue
        if operation != "add" and operation != "rm":
            print("Unexpected operation: expected 'add' or 'rm'")
            continue

        # perform command
        if operation == "add":
            if variable == "all":
                print("cannot add all variables to single axis")
                continue
            var_map[axis].add(variable)
        if operation == "rm":
            if variable == "all":
                var_map[axis].clear()
                continue
            var_map[axis].remove(variable)


def clear_variables():
    x_vars.clear()
    y_vars.clear()
    z_vars.clear()
    print("Variables Cleared")


def create_variables():
    # want to abstract this out to prewritten classes...
    # can use console to execute command line commands... maybe call script?

    # or just implement a class that extends AbstractCustomVariable
    #   - input (list of strings of input variables)
    #   - function (returns the desired value, using inputs)
    # script takes list of custom var classes
    # either applies them to every episode and store in pandas column* OR
    #  applies them to selected episode to populate a custom_var dict with Series
    print("SOL bruh")


def display_variables():
    print("Current Variables:\n\t\t x - {}\n\t\t y - {}\n\t\t z - {}".format(x_vars, y_vars, z_vars))


def plot_variables():
    global x_vars, y_vars, z_vars


menu = ConsoleMenu("AFRL RTA - Log Analysis Tool", "Enter a number from the list below:")

# Create some items
"""
    + sub plots? (whats so special about sub plots / why not just make 2 plots / how to implement?)
    + manipulate data (invoke lambda expression on plot vars for more complex analysis...)
"""

metadata_item = FunctionItem("View Metadata Table", display_metadata)

episode_menu = FunctionItem("Select Episode", set_selected_episode)

display_variables_item = FunctionItem("Display Current Variables", display_variables)
clear_variables_item = FunctionItem("Clear variables", clear_variables)
set_variables_item = FunctionItem("Set or Remove Variables", manipulate_variables)
create_variables_item = FunctionItem("Create Custom Variables", create_variables)
plot_variables_item = FunctionItem("Plot Variables", plot_variables)
graph_menu = ConsoleMenu("Graph Maker")

graph_menu_item = SubmenuItem("Make a Graph", graph_menu, menu)

# Once we're done creating them, we just add the items to the menu
graph_menu.append_item(display_variables_item)
graph_menu.append_item(set_variables_item)
graph_menu.append_item(create_variables_item)
graph_menu.append_item(plot_variables_item)

menu.append_item(metadata_item)
menu.append_item(episode_menu)
menu.append_item(graph_menu_item)

# Finally, we call show to show the menu and allow the user to interact
menu.show()





"""
BACKLOG:

abstract user input (good sys to handle input / execution of queries)?
    - config file (path to new log or path to saved dataframe, variables of interest, episode of interest [ID / ep #])
    - PTUI -> series of cmds (show metadata log_file, plot x y episode_ID, etc)*

Reduce start up time:
    look into saving pandas tables
        -
    consider HDF5

add min distance to lead, max rejoin time, reward (total?), etc to metadata table


COMPLETE:
### inconsistency found in logs -> first episode in log will have state for step zero & info will be null... ###
### can just make a dict w/ column name -> value lists kvps... make pandas table in one step* ###
### formatted table correctly ###
### find root of run issue - formatted data into dict (remove df.append ops) ###
### display metadata table for user ###
### generate t-var plot for user ###
### pickle serialization for debugging load time reduction ###
"""