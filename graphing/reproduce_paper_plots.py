import argparse
from ray.tune import Analysis
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pdb




def get_args():
    parser = argparse.ArgumentParser(description='reproduce paper plot script')
    parser.add_argument('--results_dir',type=str,default=None)
    args = parser.parse_args()
    return args

def make_plots(args):
    results_dir = args.results_dir
    all_subdirs = next(os.walk(results_dir))[1]

    # preprocessing logic

    # keep track of all csv files
    csv_file_tracker = []
    for d in all_subdirs:
        if d == 'training_logs':
            continue
        else:
            csv_path = logdir + '/' + d + '/' + 'progress.csv'
            csv_file_tracker.append(csv_path)

    data_dfs = [pd.read_csv(csv_file_tracker[i]) for i in range(len(csv_file_tracker))]


    # perform necessary timestep clipping

    # first need to ensure the following: are all timesteps equal if so - no need for clipping
    perform_clipping = False
    i = 0
    max_timestep = 0
    for ds in data_dfs:
        check_pos = ds.shape[0] -1
        current_max_timesteps = ds.iloc[[check_pos]]['timesteps_total']
        if i == 0:
            max_timestep = current_max_timesteps
        else:
            if max_timestep != current_max_timesteps:
                perform_clipping = True
                break

    perform_extrapolate = False

    if perform_clipping:
        perform_extrapolate = True
        # figure out which data_dfs did not have the same time ranges
        # then figure out which one to set as the new
        ds_ids = []
        for i,ds in enumerate(data_dfs):
            check_pos = ds.shape[0]
            current_max_timesteps = ds.iloc[[check_pos]]['timesteps_total']
            if current_max_timesteps != max_timestep:
                ds_ids.append(i)

        # find max id amongst the data points that do not
        max_id_num = -1
        max_id_timestep = -1
        for id in ds_ids:
            ds = data_dfs[id]
            check_pos = ds.shape[0] - 1
            current_max_timesteps = ds.iloc[[check_pos]]['timesteps_total']
            if current_max_timesteps > max_id_timestep:
                max_id_timestep = current_max_timesteps
                max_id_num = id






    return


def main():
    args = get_args()
    make_plots(args)


if __name__ == '__main__':
    main()
