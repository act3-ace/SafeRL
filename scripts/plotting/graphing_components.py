import os
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.interpolate as interpolate


def data_frame_processing(logdir):
    results_dir = logdir
    all_subdirs = next(os.walk(results_dir))[1]

    csv_file_tracker = []
    for d in all_subdirs:
        if d == 'training_logs':
            continue
        else:
            csv_path = results_dir + '/' + d + '/' + 'progress.csv'
            csv_file_tracker.append(csv_path)

    data_dfs = [pd.read_csv(csv_file_tracker[i]) for i in range(len(csv_file_tracker))]

    return data_dfs


def find_quantity_handle(quantity, data_dfs):
    look_quantity = quantity.lower()
    quantity_handle = None
    col_names = list(data_dfs[0].columns)

    for label in col_names:
        if look_quantity in label:
            quantity_handle = label

    if quantity_handle is None:
        print("Fatal: unable to find quantity in data files, please make sure the quantity is spelled correctly")
        print('quantity name', quantity)
        print(col_names)
        exit()

    return quantity_handle


def clip(q1, data_dfs, clip_method):
        """
        Parameters:
            q1 : str
                name of quantity to be clipped
            data_dfs : list of pandas dataframes
                a list containing all the pandas dataframes that comprise the dataset
            clip_method : str or int or tuple of ints
                If 'string' , value must be 'short_traj' to specify clipping to the minimum bound of a quantity
                If 'int' , quantity will be clipped will occur from (0,clip_method+1) ,
                If a tuple of ints is passed quantity will be clipped into following range (lower_bound,upper_bound)

        Returns
            clipped_quantity: list
                list of ints over the specified clipping bounds for a quantity

        """

        clipped_quantity = None
        # need q1 handle
        q1_handle = find_quantity_handle(q1, data_dfs)

        if clip_method == 'short_traj':
            check_pos = data_dfs[0].shape[0] - 1
            per_df_max_vals = [ds.iloc[[check_pos]][q1_handle] for ds in data_dfs]
            min_df_pos = np.argmin(per_df_max_vals)

            # give me the shortest trajectory of the specified quantity
            q_short_traj = data_dfs[min_df_pos][q1_handle]

            # set value
            clipped_quantity = np.array(q_short_traj)

        elif type(clip_method) == int:
            # clip to an upper bound
            upper_bound = clip_method

            # find longest trajectory
            check_pos = data_dfs[0].shape[0] - 1
            per_df_max_vals = [ds.iloc[[check_pos]][q1_handle] for ds in data_dfs]
            max_df_pos = np.argmax(per_df_max_vals)

            longest_trajectory_of_quantity = np.array(data_dfs[max_df_pos][q1_handle])

            # now using longest trajectory of the quantity clip it to the new range
            new_clipped_range = longest_trajectory_of_quantity[longest_trajectory_of_quantity < clip_method]

            clipped_quantity = new_clipped_range
        elif type(clip_method) == tuple:
            # clip between bounds
            bounds = clip_method
            lower_bound, upper_bound = bounds

            # find longest trajectory
            check_pos = data_dfs[0].shape[0] - 1
            per_df_max_vals = [ds.iloc[[check_pos]][q1_handle] for ds in data_dfs]
            max_df_pos = np.argmax(per_df_max_vals)

            longest_trajectory_of_quantity = np.array(data_dfs[max_df_pos][q1_handle])

            mask = np.logical_and(longest_trajectory_of_quantity < upper_bound, longest_trajectory_of_quantity > lower_bound)

            new_clipped_range = longest_trajectory_of_quantity[mask]
            clipped_quantity = new_clipped_range

        return clipped_quantity


def plot_quantity1_v_quantity2(data_dfs, q1, q2, clip_method):
    # need to look through df.columns to get appropriate handle
    q1_handle = find_quantity_handle(q1, data_dfs)
    q2_handle = find_quantity_handle(q2, data_dfs)

    interp = False
    if clip_method is not None:
        interp = True
        q1_clipped = clip(q1, data_dfs, clip_method)

    q1_track = []
    q2_track = []

    for ds in data_dfs:
        q1_value = ds[q1_handle]
        q2_value = ds[q2_handle]

        if interp:
            func_q1_v_q2 = interpolate.interp1d(q1_value, q2_value, fill_value='extrapolate')

            q2_value = func_q1_v_q2(q1_clipped)

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

    sns.set_theme()
    plot = sns.relplot(data=graph_data, x=q1, y=q2, kind='line')
    plot.set_axis_labels(q1, q2)

    # save figure here itself
    save_file = q1 + '_v_' + q2 + '.png'
    plot.savefig(save_file, dpi=1200)

    return plot


# graph wrapper function
def graph_q1_v_q2(logdir, q1, q2, clip_method):
    data_dfs = data_frame_processing(logdir)
    plot_handle = plot_quantity1_v_quantity2(data_dfs, q1, q2, clip_method)
    return plot_handle
