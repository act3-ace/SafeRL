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
    # additional args - task := docking or rejoin , specifies the keys you want to use

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
            csv_path = results_dir + '/' + d + '/' + 'progress.csv'
            csv_file_tracker.append(csv_path)

    data_dfs = [pd.read_csv(csv_file_tracker[i]) for i in range(len(csv_file_tracker))]


    # perform necessary timestep clipping -- maybe break into separate method ?

    # perform clipping - START
    # first need to ensure the following: are all timesteps equal if so - no need for clipping
    # theres something wrong here FIX IT !, maybe use the max function ? 
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

    clipped_time_range = None
    if perform_clipping:
        print("Setup clipping")
        perform_extrapolate = True
        # figure out which data_dfs did not have the same time ranges
        # then figure out which one to set as the new
        ds_ids = []
        for i,ds in enumerate(data_dfs):
            check_pos = ds.shape[0]
            current_max_timesteps = ds.iloc[[check_pos]]['timesteps_total']
            if current_max_timesteps != max_timestep:
                ds_ids.append(i)

        # find max id amongst the data points that do not extend to full range
        max_id_num = -1
        max_id_timestep = -1
        for id in ds_ids:
            ds = data_dfs[id]
            check_pos = ds.shape[0] - 1
            current_max_timesteps = ds.iloc[[check_pos]]['timesteps_total']
            if current_max_timesteps > max_id_timestep:
                max_id_timestep = current_max_timesteps
                max_id_num = id

        clipped_time_range = data_dfs[max_id_num]['timesteps_total']

    # DO CLIPPPING - END 
        
    if perform_extrapolate:
        print("doing clip & extrapolate")
        timesteps_total_track = []
        episode_len_mean_track = []
        success_mean_track = []
        eps_reward_mean_track = []

        # dont need to walk through step by step , can just grab columns
        for ds in data_dfs:
            timestep_total = ds[key_timesteps]
            episode_len_mean = ds[key_eps_len_mean]
            success_mean = ds[key_success_mean]
            reward_mean = ds[key_eps_reward_mean]

            #interpolation functions
            func_time_v_eps_len = interpolate.interp1d(timestep_total,episode_len_mean,fill_value='extrapolate')
            func_time_v_success = interpolate.interp1d(timestep_total,success_mean,fill_value='extrapolate')
            func_time_v_reward = interpolate.interp1d(timestep_total,reward_mean,fill_value='extrapolate')
            interp_eps_len = func_time_v_eps_len(clipped_timesteps)
            interp_success = func_time_v_success(clipped_timesteps)
            interp_reward = func_time_v_reward(clipped_timesteps)

            for i in clipped_timesteps:
                timesteps_total_track.append(i)
            for i in interp_eps_len:
                episode_len_mean_track.append(i)
            for i in interp_success:
                success_mean_track.append(i)
            for i in interp_reward:
                eps_reward_mean_track.append(i)

        sns.set_theme()
        sns.set(font_scale=1.5)
        timesteps_total_v_episode_len_mean = pd.DataFrame()
        timesteps_total_v_episode_len_mean[key_timesteps] = timesteps_total_track
        timesteps_total_v_episode_len_mean[key_eps_len_mean] = episode_len_mean_track

        timesteps_total_v_success_mean = pd.DataFrame()
        timesteps_total_v_success_mean[key_timesteps] = timesteps_total_track
        timesteps_total_v_success_mean['success_mean'] = success_mean_track

        timesteps_total_v_episode_reward_mean = pd.DataFrame()
        timesteps_total_v_episode_reward_mean[key_timesteps] = timesteps_total_track
        timesteps_total_v_episode_reward_mean['episode_reward_mean'] = eps_reward_mean_track

        success_mean_plot = sns.relplot(data=timesteps_total_v_success_mean,x='timesteps_total',y='success_mean',kind='line')
        success_mean_plot.set_axis_labels("Timesteps","Success Rate")

        episode_mean_len_plot = sns.relplot(data=timesteps_total_v_episode_len_mean,x='timesteps_total',y='episode_len_mean',kind='line')
        episode_mean_len_plot.set_axis_labels("Timesteps","Episode Length")

        reward_plot = sns.relplot(data=timesteps_total_v_episode_reward_mean,x='timesteps_total',y='episode_reward_mean',kind='line')
        reward_plot.set_axis_labels("Timesteps","Average Return")

        episode_mean_len_plot.savefig('docking2d_eps_len_plot.png',dpi=1200)
        success_mean_plot.savefig('docking2d_success_mean.png',dpi=1200)
        reward_plot.savefig('docking2d_reward_graph.png',dpi=1200)

    else:
        print("Normal procedure")
        # prepare plots for the typical range e.g. eps_len, success_mean,reward_mean,return_graph
        key_timesteps = 'timesteps_total'
        key_eps_len_mean = 'episode_len_mean'
        key_success_mean = 'custom_metrics/outcome/success_mean'
        key_eps_reward_mean = 'episode_reward_mean'

        timesteps_total_track = []
        episode_len_mean_track = []
        success_mean_track = []
        eps_reward_mean_track = []

        for ds in data_dfs:
            timestep_total = ds[key_timesteps]
            episode_len_mean = ds[key_eps_len_mean]
            success_mean = ds[key_success_mean]
            reward_mean = ds[key_eps_reward_mean]

            for i in timestep_total:
                timesteps_total_track.append(i)
            for i in episode_len_mean:
                episode_len_mean_track.append(i)
            for i in success_mean:
                success_mean_track.append(i)
            for i in reward_mean:
                eps_reward_mean_track.append(i)

        sns.set_theme()
        sns.set(font_scale=1.5)
        timesteps_total_v_episode_len_mean = pd.DataFrame()
        timesteps_total_v_episode_len_mean[key_timesteps] = timesteps_total_track
        timesteps_total_v_episode_len_mean[key_eps_len_mean] = episode_len_mean_track

        timesteps_total_v_success_mean = pd.DataFrame()
        timesteps_total_v_success_mean[key_timesteps] = timesteps_total_track
        timesteps_total_v_success_mean['success_mean'] = success_mean_track

        timesteps_total_v_episode_reward_mean = pd.DataFrame()
        timesteps_total_v_episode_reward_mean[key_timesteps] = timesteps_total_track
        timesteps_total_v_episode_reward_mean['episode_reward_mean'] = eps_reward_mean_track

        success_mean_plot = sns.relplot(data=timesteps_total_v_success_mean,x='timesteps_total',y='success_mean',kind='line')
        success_mean_plot.set_axis_labels("Timesteps","Success Rate")

        episode_mean_len_plot = sns.relplot(data=timesteps_total_v_episode_len_mean,x='timesteps_total',y='episode_len_mean',kind='line')
        episode_mean_len_plot.set_axis_labels("Timesteps","Episode Length")

        reward_plot = sns.relplot(data=timesteps_total_v_episode_reward_mean,x='timesteps_total',y='episode_reward_mean',kind='line')
        reward_plot.set_axis_labels("Timesteps","Average Return")

        episode_mean_len_plot.savefig('docking2d_eps_len_plot.png',dpi=1200)
        success_mean_plot.savefig('docking2d_success_mean.png',dpi=1200)
        reward_plot.savefig('docking2d_reward_graph.png',dpi=1200)

    return


def main():
    args = get_args()
    make_plots(args)


if __name__ == '__main__':
    main()
