import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import numbers


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

        if clip_method == 'shortest':
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
            per_df_max_vals = [ds.iloc[[-1]][q1_handle] for ds in data_dfs]
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

def get_clip_bounds(label, data_dfs, clip_method):
    """
    Parameters:
        label : str
            name of quantity to be clipped
        data_dfs : list of pandas dataframes
            a list containing all the pandas dataframes that comprise the dataset
        clip_method : str or int or tuple of ints
            If 'string' , value must be 'shortest' to specify clipping to the minimum bound of a quantity
            If 'int' , quantity will be clipped will occur from (0,clip_method+1) ,
            If a tuple of ints is passed quantity will be clipped into following range (lower_bound,upper_bound)

    Returns
        clip_bounds: tuple
            min and max clipping boundaries

    """

    clip_min = -np.inf
    clip_max = np.inf

    if clip_method == 'shortest':
        clip_max = np.min([np.max(df[label]) for df in data_dfs])
    elif isinstance(clip_method, numbers.Number):
        clip_max = clip_method
    elif type(clip_method) == tuple:
        assert len(clip_method) == 2, "For tuple clipping bounds, the tuple must be length 2"
        clip_min, clip_max = clip_method

    return clip_min, clip_max

def clip_array(arr, bounds):
    clip_idx = np.logical_and(arr >= bounds[0], arr <= bounds[1])

    return arr[clip_idx], clip_idx

def plot_quantity1_v_quantity2(data_dfs, q1, q2, clip_method, output_dir='./', x_label=None, y_label=None, interp=True, interp_subsample_len=None, rc_params={}):
    assert interp, "Not currently implemented without interp parameter"
    if x_label is None:
        x_label = q1
    if y_label is None:
        y_label = q2

    # need to look through df.columns to get appropriate handle
    q1_handle = find_quantity_handle(q1, data_dfs)
    q2_handle = find_quantity_handle(q2, data_dfs)

    clip_bounds = get_clip_bounds(q1_handle, data_dfs, clip_method)

    q1_tracks = []
    q2_tracks = []

    # define q1 interp points:
    if interp:
        q1_interp = np.sort(np.unique(np.concatenate([df[q1_handle].to_numpy() for df in data_dfs])))
        q1_interp, _ = clip_array(q1_interp, clip_bounds)

        if interp_subsample_len is not None:
            q1_interp = np.linspace(q1_interp[0], q1_interp[-1], interp_subsample_len)


    for ds in data_dfs:
        q1_value = ds[q1_handle].to_numpy()
        q2_value = ds[q2_handle].to_numpy()
        

        if interp:
            func_q1_v_q2 = interpolate.interp1d(q1_value, q2_value, bounds_error=False, fill_value=(q2_value[0], q2_value[-1]))
            
            q1_clipped = q1_interp
            q2_clipped = func_q1_v_q2(q1_clipped)
        else:
            q1_clipped, clip_idx = clip_array(q1_value, clip_bounds)

            q1_clipped = q1_value[clipped_idx]
            q2_clipped = q2_value[clipped_idx]

        q1_tracks += [q1_clipped]
        q2_tracks += [q2_clipped]

    sns.set_theme(rc=rc_params)

    plot = sns.relplot(x=np.concatenate(q1_tracks), y=np.concatenate(q2_tracks), kind='line', height=rc_params['figure.figsize'][1])
    plot.ax.set_xlabel(x_label, fontstyle='italic')
    plot.ax.set_ylabel(y_label, fontstyle='italic')
    
    plot.fig.set_size_inches(rc_params['figure.figsize'][0], rc_params['figure.figsize'][1])
    

    # save figure here itself
    save_file = os.path.join(output_dir, q1 + '_v_' + q2 + '.png')
    plot.savefig(save_file, bbox_inches="tight", pad_inches=0.04)

    return plot, q1_tracks, q2_tracks


# graph wrapper function
def graph_q1_v_q2(logdir, q1, q2, clip_method, rename_map=None, **kwargs):
    data_dfs = data_frame_processing(logdir)
    plot_handle = plot_quantity1_v_quantity2(data_dfs, q1, q2, clip_method, x_label=rename_map.get(q1, q1), y_label=rename_map.get(q2, q2), **kwargs)
    return plot_handle

def compute_interaction_efficiency(timesteps_list, success_list, threshold=0.8):
    timesteps = np.stack(timesteps_list,axis=0)
    success = np.stack(success_list,axis=0)

    assert np.all(timesteps == timesteps_list[0]), "all trajectories must have their timesteps interpolated to equal values. Please use interp option when composing data"

    success_avg = np.mean(success, axis=0)
    
    inter_eff = timesteps[0, success_avg >= threshold][0]

    return inter_eff