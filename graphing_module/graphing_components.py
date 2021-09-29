import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.interpolate as interpolate

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

# how should clip_method be specified ?
def perform_timestep_clipping(clip_method,data_dfs):
    # 3 cases , clip to upper bound , clip to shortest trajectory, specified trajectory
    clipped_timesteps = None

    # fine
    if args.clip_short_traj:
        check_pos = data_dfs[0].shape[0] -1
        per_df_max_timesteps = [ds.iloc[[check_pos]]['timesteps_total'] for ds in data_dfs]
        min_df_total_timesteps = np.argmin(per_df_max_timesteps)

        check_max_pos_row = data_dfs[min_df_total_timesteps].shape[0]-1
        timestep_total_min = data_dfs[min_df_total_timesteps].iloc[[check_max_pos_row]]['timesteps_total']
        clipped_timesteps = np.array(list(range(timestep_total_min)))
    # needs its own argument
    elif args.clip_to_upper_bound != None:
        upper_bound = args.clip_to_upper_bound
        clipped_timesteps = np.array(list(range(int(upper_bound)+1)))
    # needs
    elif args.clip_to_bounds != None:
        bounds = args.clip_to_bounds
        lower_bound, upper_bound = bounds
        clipped_timesteps = np.array(list(range(lower, upper_bound+1)))

    return clipped_timesteps

def find_quantity_handle(quantity,data_dfs):
    look_quantity = quantity.lower()
    quantity_handle = None
    for label in col_names:
        if look_quantity in label:
            quantity_handle = label

    if look_quantity == None:
        print("Fatal: unable to find quantity in data files, please make sure the quantity is spelled correctly")
        exit()

    return quantity_handle

# make a general clipping function
# make a general plot function , x vs y

def general_clip():
    return

def plot_quantity1_v_quantity2():
    return

def plot_timesteps_v_quantity(data_dfs,quantity,exper_name):
    quantity_handle = find_quantity_handle(quantity,data_dfs)
    # need to look through df.columns to get appropriate handle

    timesteps_total_track = []
    quantity_track = []

    for ds in data_dfs:
        timestep_total = ds['timesteps_total']
        quantity = ds[quantity_handle]

        for i in timestep_total:
            timesteps_total_track.append(i)
        for i in quantity:
            quantity_track.append(i)

    graph_data = pd.DataFrame()
    graph_data['timesteps'] = timesteps_total_track
    graph_data[quantity] = episode_len_mean_track

    plot = sns.relplot(data=graph_data,x='timesteps',y=quantity,kind='line')
    plot.set_axis_labels("Timesteps",quantity)

    # save figure here itself
    save_file = exper_name + 'timesteps' + '_v_' + quantity + '.png'
    plot.savefig(save_file,dpi=1200)

    return plot

# this function is used alongside clipping,
# timesteps should be set based on the clipping needed
def plot_interp_time_v_quantity(data_dfs,quantity,exper_name,clipped_timesteps):
    quantity_handle = find_quantity_handle(quantity,data_dfs)
    # need to look through df.columns to get appropriate handle

    timesteps_total_track = []
    quantity_track = []

    for ds in data_dfs:
        timestep_total = ds['timesteps_total']
        quantity = ds[quantity_handle]

        func_time_v_quantity = interpolate.interp1d(timestep_total,quantity,fill_value='extrapolate')

        interp_quantity = func_time_v_quantity(clipped_timesteps)

        for i in clipped_timesteps:
            timesteps_total_track.append(i)
        for i in interp_quantity:
            quantity_track.append(i)

    graph_data = pd.DataFrame()
    graph_data['timesteps'] = timesteps_total_track
    graph_data[quantity] = quantity_track

    plot = sns.relplot(data=graph_data,x='timesteps',y=quantity,kind='line')
    plot.set_axis_labels("Timesteps",quantity)

    # save figure here itself
    save_file = exper_name + 'timesteps' + '_v_' + quantity + '.png'
    plot.savefig(save_file,dpi=1200)

    return plot


# general graph function
# needs logdir
# set of quantities
# pass clipping method
def graph():
    return
