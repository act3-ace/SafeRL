import argparse
from ray.tune import Analysis
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pdb
import scipy.interpolate as interpolate


def get_args():
    # additional args - task := docking or rejoin , specifies the keys you want to use

    parser = argparse.ArgumentParser(description='reproduce training curve plots')
    parser.add_argument('--results_dir',type=str,default=None)

     # are you using clipping at all, if false no clippping
    parser.add_argument('--clip',type=bool,default=False)

    # clip to shortest trajectory
    parser.add_argument('--clip_short_traj',type=bool)

    # numeric clipping bound, upper bound
    parser.add_argument('--clip_to_upper_bound',type=float,help='upper bound to clip to')

    # bounded clipping
    parser.add_argument('--clip_to_interval',type=list,help='list containing upper and lower bound to clip to')

    args = parser.parse_args()

    # run a check
    # in order to use clipping method, clip must be true,
    # if clip is True one of the three must be set , not all

    if args.clip == True:
        if (args.clip_short_traj == None) and (args.clip_to_upper_bound == None) and (args.clip_to_interval == None):
            parser.error(' "clip" was set to true , but no clipping procedure was selected')
    elif args.clip == False:
        if (args.clip_short_traj != None) or (args.clip_to_upper_bound != None) or (args.clip_to_interval != None):
            parser.error(' "clip" was set to false, therefore a clipping procedure cannot be selected')


    return args


def data_frame_processing(args):
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

    check_dir = all_subdirs[0]
    check_dir_split = check_dir.lower().split("_")

    rejoin_check = any('rejoin' in s for s in check_dir_split)
    docking_check = any('docking' in s for s in check_dir_split)

    exper_name = None
    if rejoin_check == True:
        exper_name = 'rejoin'
    elif docking_check == True:
        exper_name = 'docking'

    return data_dfs, exper_name

# is clipping required for the dataset ?
def determine_clipping(args,data_dfs):
    # do we need to clip ?
    need_clipping = False
    check_pos = data_dfs[0].shape[0] -1
    per_df_max_timesteps = [ds.iloc[[check_pos]]['timesteps_total'] for ds in data_dfs]
    max_df_total_timesteps = np.argmax(per_df_max_timesteps)
    min_df_total_timesteps = np.argmin(per_df_max_timesteps)

    if max_df_total_timesteps == min_df_total_timesteps:
        need_clipping = False
    else:
        need_clipping = True

    return need_clipping


def do_clipping(args,data_dfs):
    # 3 cases , clip to upper bound , clip to shortest trajectory, specified trajectory
    clipped_timesteps = None

    if args.clip_short_traj:
        check_pos = data_dfs[0].shape[0] -1
        per_df_max_timesteps = [ds.iloc[[check_pos]]['timesteps_total'] for ds in data_dfs]
        min_df_total_timesteps = np.argmin(per_df_max_timesteps)
        check_max_pos_row = data_dfs[min_df_total_timesteps].shape[0]-1
        timestep_total_min = data_dfs[min_df_total_timesteps].iloc[[check_max_pos_row]]['timesteps_total']
        clipped_timesteps = np.array(list(range(timestep_total_min)))
    elif args.clip_to_upper_bound != None:
        upper_bound = args.clip_to_upper_bound
        clipped_timesteps = np.array(list(range(int(upper_bound)+1)))
    elif args.clip_to_bounds != None:
        bounds = args.clip_to_bounds
        lower_bound, upper_bound = bounds
        clipped_timesteps = np.array(list(range(lower, upper_bound+1)))

    return clipped_timesteps


def graph_rejoin(args,data_dfs,rejoin_keys):

    print("Graphing Rejoin Experiment")


    timesteps_total_track = []
    episode_len_mean_track = []
    success_mean_track = []
    eps_reward_mean_track = []

    print(args.clip)

    if args.clip:
        clipped_timesteps = do_clipping(args,data_dfs)

        for ds in data_dfs:
            timestep_total = ds[rejoin_keys['key_timesteps']]
            episode_len_mean = ds[rejoin_keys['key_eps_len_mean']]
            success_mean = ds[rejoin_keys['key_success_mean']]
            reward_mean = ds[rejoin_keys['key_eps_reward_mean']]

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

    else:

        for ds in data_dfs:
            timestep_total = ds[rejoin_keys['key_timesteps']]
            episode_len_mean = ds[rejoin_keys['key_eps_len_mean']]
            success_mean = ds[rejoin_keys['key_success_mean']]
            reward_mean = ds[rejoin_keys['key_eps_reward_mean']]

            for i in timestep_total:
                timesteps_total_track.append(i)
            for i in episode_len_mean:
                episode_len_mean_track.append(i)
            for i in success_mean:
                success_mean_track.append(i)
            for i in reward_mean:
                eps_reward_mean_track.append(i)

    print("Data Aggregration complete ")

    sns.set_theme()
    sns.set(font_scale=1.5)

    print('setup timesteps vs episode_len_mean')
    timesteps_total_v_episode_len_mean = pd.DataFrame()
    timesteps_total_v_episode_len_mean[key_timesteps] = timesteps_total_track
    timesteps_total_v_episode_len_mean['episode_len_mean'] = episode_len_mean_track
    print('done')

    print('setup timesteps vs success_mean')
    timesteps_total_v_success_mean = pd.DataFrame()
    timesteps_total_v_success_mean[key_timesteps] = timesteps_total_track
    timesteps_total_v_success_mean['success_mean'] = success_mean_track
    print("done")

    print("setup timestep_total vs episode_reward_mean")
    timesteps_total_v_episode_reward_mean = pd.DataFrame()
    timesteps_total_v_episode_reward_mean[key_timesteps] = timesteps_total_track
    timesteps_total_v_episode_reward_mean['episode_reward_mean'] = eps_reward_mean_track
    print("done")

    print("plot timesteps vs success_mean")
    success_mean_plot = sns.relplot(data=timesteps_total_v_success_mean,x='timesteps_total',y='success_mean',kind='line')
    success_mean_plot.set_axis_labels("Timesteps","Success Rate")
    print('done')

    print("plot timesteps vs episode_len_mean")
    episode_mean_len_plot = sns.relplot(data=timesteps_total_v_episode_len_mean,x='timesteps_total',y='episode_len_mean',kind='line')
    episode_mean_len_plot.set_axis_labels("Timesteps","Episode Length")
    print("done")

    print("plot timesteps vs average return")
    reward_plot = sns.relplot(data=timesteps_total_v_episode_reward_mean,x='timesteps_total',y='episode_reward_mean',kind='line')
    reward_plot.set_axis_labels("Timesteps","Average Return")
    print('done')

    print("saving all graphs to files ")
    episode_mean_len_plot.savefig('rejoin_eps_len_plot.png',dpi=1200)
    success_mean_plot.savefig('rejoin_success_mean.png',dpi=1200)
    reward_plot.savefig('rejoin_reward_graph.png',dpi=1200)


def graph_docking(args,data_dfs,docking_keys):

    print("Graphing Docking Experiment")


    timesteps_total_track = []
    episode_len_mean_track = []
    success_mean_track = []
    eps_reward_mean_track = []
    const_viol_ratio_track = []
    delta_v_track = []


    if args.clip:
        clipped_timesteps = do_clipping(args,data_dfs)

        for ds in data_dfs:
            timestep_total = ds[docking_keys['key_timesteps']]
            episode_len_mean = ds[docking_keys['key_eps_len_mean']]
            success_mean = ds[docking_keys['key_success_mean']]
            reward_mean = ds[docking_keys['key_eps_reward_mean']]
            const_viol_ratio = ds[docking_keys['key_const_viol']]
            delta_v_mean = ds[docking_keys['key_delta_v']]

            func_time_v_eps_len = interpolate.interp1d(timestep_total,episode_len_mean,fill_value='extrapolate')
            func_time_v_success = interpolate.interp1d(timestep_total,success_mean,fill_value='extrapolate')
            func_time_v_reward = interpolate.interp1d(timestep_total,reward_mean,fill_value='extrapolate')
            func_time_v_constr_viol = interpolate.interp1d(timestep_total,const_viol_ratio,fill_value='extrapolate')
            func_time_v_deltav = interpolate.interp1d(timestep_total,delta_v_mean,fill_value='extrapolate')

            interp_delta_v = func_time_v_deltav(clipped_timesteps)
            interp_constr_viol = func_time_v_constr_viol(clipped_timesteps)
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
            for i in interp_constr_viol:
                const_viol_ratio_track.append(i)
            for i in interp_delta_v:
                delta_v_track.append(i)


    else:

        for ds in data_dfs:
            timestep_total = ds[rejoin_keys['key_timesteps']]
            episode_len_mean = ds[rejoin_keys['key_eps_len_mean']]
            success_mean = ds[rejoin_keys['key_success_mean']]
            reward_mean = ds[rejoin_keys['key_eps_reward_mean']]
            const_viol_ratio = ds[docking_keys['key_const_viol']]
            delta_v_mean = ds[docking_keys['key_delta_v']]


            for i in timestep_total:
                timesteps_total_track.append(i)
            for i in episode_len_mean:
                episode_len_mean_track.append(i)
            for i in success_mean:
                success_mean_track.append(i)
            for i in reward_mean:
                eps_reward_mean_track.append(i)
            for i in constr_viol_ratio:
                const_viol_ratio_track.append(i)
            for i in delta_v_mean:
                delta_v_track.append(i)

    print('Data Aggregration Complete')
    print('Now plotting')

    sns.set_theme()
    sns.set(font_scale=1.5)

    print("Setup timestep_total v episode_len_mean")
    timesteps_total_v_episode_len_mean = pd.DataFrame()
    timesteps_total_v_episode_len_mean[docking_keys['key_timesteps']] = timesteps_total_track
    timesteps_total_v_episode_len_mean['episode_len_mean'] = episode_len_mean_track
    print('Done')

    print("Setup timestep_total vs success_mean")
    timesteps_total_v_success_mean = pd.DataFrame()
    timesteps_total_v_success_mean[docking_keys['key_timesteps']] = timesteps_total_track
    timesteps_total_v_success_mean['success_mean'] = success_mean_track
    print('Done')

    print("Setup timestep_total vs episode_reward_mean")
    timesteps_total_v_episode_reward_mean = pd.DataFrame()
    timesteps_total_v_episode_reward_mean[docking_keys['key_timesteps']] = timesteps_total_track
    timesteps_total_v_episode_reward_mean['episode_reward_mean'] = eps_reward_mean_track
    print("Done")

    print("Setup timestep_total vs delta_v")
    timesteps_total_v_constr_viol = pd.DataFrame()
    timesteps_total_v_constr_viol[docking_keys['key_timesteps']] = timesteps_total_track
    timesteps_total_v_constr_viol['delta_v'] = delta_v_track
    print("Done")

    print("Setup timesteps_total vs constr_viol")
    timesteps_total_v_constr_viol = pd.DataFrame()
    timesteps_total_v_constr_viol[docking_keys['key_timesteps']] = timesteps_total_track
    timesteps_total_v_constr_viol['constr_viol'] = const_viol_ratio_track

    print("plotting timesteps vs success rate")
    success_mean_plot = sns.relplot(data=timesteps_total_v_success_mean,x='timesteps_total',y='success_mean',kind='line')
    success_mean_plot.set_axis_labels("Timesteps","Success Rate")
    print("done")

    print("plotting timesteps vs episode_len_mean")
    episode_mean_len_plot = sns.relplot(data=timesteps_total_v_episode_len_mean,x='timesteps_total',y='episode_len_mean',kind='line')
    episode_mean_len_plot.set_axis_labels("Timesteps","Episode Length")
    print('done')

    print("plotting timesteps vs episode_reward_mean")
    reward_plot = sns.relplot(data=timesteps_total_v_episode_reward_mean,x='timesteps_total',y='episode_reward_mean',kind='line')
    reward_plot.set_axis_labels("Timesteps","Average Return")
    print('done')

    print("plotting timesteps vs delta_v")
    deltav_plot = sns.relplot(data=timesteps_total_v_constr_viol,x='timesteps_total',y='delta_v',kind='line')
    deltav_plot.set_axis_labels("Timesteps","Delta V")
    print('done')

    print("plotting timestep_total vs constraint violation ratio")
    constr_viol_plot = sns.relplot(data=timesteps_total_v_constr_viol,x='timesteps_total',y='constr_viol',kind='line')
    constr_viol_plot.set_axis_labels("Timesteps","Constraint Violation Ratio")
    print('done')

    print("saving all graphs to png")
    episode_mean_len_plot.savefig('docking_eps_len_plot.png',dpi=1200)
    success_mean_plot.savefig('docking_success_mean.png',dpi=1200)
    reward_plot.savefig('docking_reward_graph.png',dpi=1200)
    deltav_plot.savefig('docking_delta_v.png',dpi=1200)
    constr_viol_plot.savefig('docking_constraint_violation.png',dpi=1200)


def make_plots(args):

    docking_keys = {'key_timesteps':'timesteps_total',
                    'key_const_viol':'custom_metrics/constraint_violation.max_vel_constraint.ratio_mean',
                    'key_eps_len_mean' : 'episode_len_mean',
                    'key_success_mean' : 'custom_metrics/outcome/success_mean',
                    'key_eps_reward_mean' : 'episode_reward_mean',
                    'key_delta_v' : 'custom_metrics/delta_v_total_mean'}

    rejoin_keys = {'key_timesteps':'timesteps_total',
                    'key_eps_len_mean' : 'episode_len_mean',
                    'key_success_mean' : 'custom_metrics/outcome/success_mean',
                    'key_eps_reward_mean' : 'episode_reward_mean',}

    data_dfs ,exper_name = data_frame_processing(args)

    need_clipping = determine_clipping(args,data_dfs)

    if need_clipping and args.clip == False:
        print("Fatal: This dataset needs a clipping procedure")
        exit()

    if exper_name == 'docking':
        graph_docking(args,data_dfs,docking_keys)
    elif exper_name == 'rejoin':
        graph_rejoin(args,data_dfs,rejoin_keys)

    return

def main():
    args = get_args()
    make_plots(args)


if __name__ == '__main__':
    main()
