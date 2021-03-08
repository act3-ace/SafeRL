"""
Script to prototype a dev debugging tool that presents RLLib logs in a readable, informant and directed manner.

Note: requires flatten_json package
        (pip install flatten_json)

Author: John McCarroll
3-5-2021
"""

import pandas as pd
import numpy as np
import jsonlines
from matplotlib import pyplot
from flatten_json import flatten_json

### Consume log file and const pandas tables
path_to_log = "/home/john/AFRL/Dubins/have-deepsky/rejoin/output/expr_20210305_151007/training_logs/worker_1.log"
# worker_episode_num = 11

# TODO -> one tables dict
metadata_table = {
    "episode_ID": [],
    "episode_duration": [],
    "episode_success": [],
    "episode_failure": []

}                # + max rejoin duration, min lead dist?
episode_dictionaries = {}        # index (row #) -> dict (flattened state dict) [step_number | episode_ID | wingman_x | ... ]
blacklist = ["obs", "time"]              # list of log keys to omit from pandas table


# open log file
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
                metadata_table["episode_ID"].append(prev_ID)
                metadata_table["episode_duration"].append(episode_duration)
                metadata_table["episode_success"].append(episode_success)
                metadata_table["episode_failure"].append(episode_failure)

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

# construct episode DataFrames
episode_dataframes = {}
for ID, episode_dict in episode_dictionaries.items():
    episode_dataframes[ID] = pd.DataFrame.from_dict(episode_dict, "index")

# construct meta data FataFrame
metadata_df = pd.DataFrame(metadata_table)
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(metadata_df)

# TODO save pandas tables for faster load (HDF5? Serialize?)

### Query Data (GUI? PTUI?)
while True:
    ## get input
    # display available episode IDs
    # print()
    # print(list(episode_dataframes.keys()))
    ep_ID = int(input("Select Episode by ID:"))
    # display available variables
    for _key, val in episode_dataframes.items():
        print()
        print("variables:")
        print(val.columns.values)
        break
    col = input("Select variable of interest:")

    ## generate plot
    # collect data
    if ep_ID in episode_dataframes:
        ep_df = episode_dataframes[ep_ID]
    else:
        print("invalid episode ID")
        continue

    try:
        data = ep_df[col]                       # if col incorrect -> NameError
    except NameError:
        print("invalid variable name")
        continue

    t = ep_df["step_number"]

    # make plot
    pyplot.plot(t, data)
    pyplot.xlabel("step number")                # *change to time?
    pyplot.ylabel(col)
    pyplot.title("Episode {}:\n {} vs. Step Number".format(ep_ID, col))
    pyplot.show()

    print("done!")
    print()




"""
BACKLOG:

episode success and failure - not working**

Reduce start up time:
    look into saving pandas tables
        -
    consider HDF5
    
abstract user input (good sys to handle input / execution of queries)?
    - config file (path to new log or path to saved dataframe, variables of interest, episode of interest [ID / ep #])
    - PTUI -> series of cmds (show metadata log_file, plot x y episode_ID, etc)*

attempt full worker log?

add min distance to lead, max rejoin time, reward (total?), etc to metadata table



COMPLETE:
### inconsistency found in logs -> first episode in log will have state for step zero & info will be null... ###
### can just make a dict w/ column name -> value lists kvps... make pandas table in one step* ###
### formatted table correctly ###
### find root of run issue - formatted data into dict (remove df.append ops) ###
### display metadata table for user ###
### generate t-var plot for user ###
"""