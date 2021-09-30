import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import scipy.interpolate as interpolate

def data_frame_processing(logdir):
    results_dir = logdir
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



# how should clip_method be specified ?
def clip(q1,data_dfs,clip_method):
    # 3 cases , clip to upper bound , clip to shortest trajectory, specified trajectory
    clipped_timesteps = None
    # need q1 handle
    q1_handle = find_quantity_handle(q1,data_dfs)

    if clip_method == 'short_traj':
        check_pos = data_dfs[0].shape[0] -1
        per_df_max_timesteps = [ds.iloc[[check_pos]][q1_handle] for ds in data_dfs]
        min_df_pos = np.argmin(per_df_max_vals)

        check_max_pos_row = data_dfs[min_df_pos].shape[0]-1
        q1_min = data_dfs[min_df_pos].iloc[[check_max_pos_row]][q1_handle]
        clipped_timesteps = np.array(list(range(q1_min)))
    # needs its own argument
    elif type(clip_method) == int:
        # clip to an upper bound
        upper_bound = clip_method
        clipped_timesteps = np.array(list(range(int(upper_bound)+1)))
    elif type(clip_method) == tuple:
        bounds = clip_method
        lower_bound, upper_bound = bounds
        clipped_timesteps = np.array(list(range(lower, upper_bound+1)))

    return clipped_timesteps


def plot_quantity1_v_quantity2(data_dfs,q1,q2,clip_method):
    q1_handle = find_quantity_handle(q1,data_dfs)
    q2_handle = find_quantity_handle(q2,data_dfs)
    # need to look through df.columns to get appropriate handle
    interp = False
    if clip_method is not None:
        interp = True
        clipped_q1 = clip(clip_method,data_dfs,q1)


    q1_track = []
    q2_track = []


    for ds in data_dfs:
        q1_value = ds[q1_handle]
        q2_value = ds[quantity_handle]

        if interp:
            func_q1_v_q2 = interpolate.interp1d(q1_value,q2_value,fill_value='extrapolate')

            q2_value = func_time_v_quantity(clipped_q1)

            for i in q1_clipped:
                q1_track.append(i)
        else:
            for i in q1_value:
                q1_track.append(i)

        for i in q2_value:
            q2_track.append(i)

    graph_data = pd.DataFrame()
    graph_data[q1] = q1_track
    graph_data[q2] = q2_track

    plot = sns.relplot(data=graph_data,x=q1,y=q2,kind='line')
    plot.set_axis_labels(q1,q2)

    # save figure here itself
    save_file =  q1 + '_v_' + q2 + '.png'
    plot.savefig(save_file,dpi=1200)

    return plot

# general graph function
# needs logdir
# set of quantities
# pass clipping method
def graph_q1_v_q2(logdir,q1,q2,clip_method):
    data_dfs,exper_name = data_frame_processing(logdir)
    plot_handle = plot_quantity1_v_quantity2(data_dfs,q1,q2,clip_method)
    return plot_handle
