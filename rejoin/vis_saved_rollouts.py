import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.animation as animation
import os

import tqdm

def plot_trajectories(trajectory_data, output_filename, colormap='jet'):
    for traj in trajectory_data:
        wingman_pos = traj['wingman_pos']
        lead_pos = traj['lead_pos']

        plt.plot(wingman_pos[:,0],wingman_pos[:,1], color=fp_color)
        plt.plot(lead_pos[:,0],lead_pos[:,1], 'g')

    plt.savefig(output_filename)

def animate_trajectories(trajectory_data, output_filename, colormap='jet', anim_rate=4, trail_length=40, plot_rejoin_region=False, plot_safety_region=False, rejoin_color_type='g', sq_axis=False, extra_time=2):
    ims = []

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure()
    ax = plt.axes()

    if sq_axis:
        ax.set_aspect('equal', adjustable='box')

    max_time = max([traj['wingman_pos'].shape[0] for traj in trajectory_data])

    max_time = int(max_time + extra_time*1000/50*anim_rate)

    num_rollouts = len(trajectory_data)
    cmap = cm.get_cmap(colormap)

    for t_idx in range(0, max_time, anim_rate):
        time_artists = []
        for traj_idx, traj in enumerate(trajectory_data):
            wingman_pos = traj['wingman_pos']
            lead_pos = traj['lead_pos']
            rejoin_point_pos = traj['rejoin_point_pos']

            traj_len = wingman_pos.shape[0]
            traj_time = min(t_idx, traj_len-1)

            fp_color = cmap(1 - traj_idx/num_rollouts)

            trail_start = max(t_idx-trail_length, 0)
            wingman_trail = wingman_pos[trail_start:(traj_time+1),:]

            if rejoin_color_type == 'match':
                lead_color = fp_color
            else:
                lead_color = 'g'

            lead_marker = 'p'

            if traj['success'][traj_time]:
                wingman_marker = '*'
            elif traj['failure'][traj_time] == 'failure_distance':
                wingman_marker = '4'
            elif traj['failure'][traj_time] == 'failure_timeout':
                wingman_marker = '+'
            elif traj['failure'][traj_time] == 'failure_crash':
                wingman_marker = 'x'
                lead_marker = 'x'
            else:
                wingman_marker = '^'

            

            wingman_plot = plt.plot(wingman_pos[traj_time,0],wingman_pos[traj_time,1], color=fp_color, marker=wingman_marker)
            lead_plot = plt.plot(lead_pos[traj_time,0],lead_pos[traj_time,1], color=lead_color, marker=lead_marker)

            wingman_trail_plot = plt.plot(wingman_trail[:,0], wingman_trail[:,1], color=fp_color)

            if plot_rejoin_region or plot_safety_region:
                death_radius = 100
                rejoin_max_radius = 100

                rejoin_region_artists = []
                # rejoin_region_artists.append(plt.Circle((lead_pos[traj_time,0], lead_pos[traj_time,1]), rejoin_max_radius, color='g'))
                
                # rejoin_region_artists.append(plt.Circle((lead_pos[traj_time,0], lead_pos[traj_time,1]), rejoin_min_radius, color='w'))
                if plot_rejoin_region:
                    if rejoin_color_type == 'match':
                        rejoin_color = fp_color
                    else:
                        rejoin_color = rejoin_color_type
                    rejoin_region_artists.append(plt.Circle((rejoin_point_pos[traj_time,0], rejoin_point_pos[traj_time,1]), rejoin_max_radius, color=rejoin_color, alpha=0.5))

                if plot_safety_region:
                    rejoin_region_artists.append(plt.Circle((lead_pos[traj_time,0], lead_pos[traj_time,1]), death_radius, color='r', alpha=0.5))


                for circle_artist in rejoin_region_artists:
                    ax.add_patch(circle_artist)

                rejoin_region_artists += plt.plot(rejoin_point_pos[traj_time,0], rejoin_point_pos[traj_time,1],  'k--')

                time_artists += rejoin_region_artists


            time_artists += wingman_plot
            time_artists += lead_plot
            time_artists += wingman_trail_plot

        ims.append(time_artists)

    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                   blit=True)
    im_ani.save(output_filename, writer=writer)

def process_rollout_data(rollout_seq):
    trajectory_data = []

    num_rollouts = len(rollout_seq)

    for rollout_idx, rollout_data in tqdm.tqdm(enumerate(rollout_seq), total=len(rollout_seq)):
        wingman_pos = np.zeros((len(rollout_data['info_history']), 2))
        lead_pos = np.zeros((len(rollout_data['info_history']), 2))
        rejoin_point_pos = np.zeros((len(rollout_data['info_history']), 2))

        success_list = []
        failure_list = []

        for t_idx, info in enumerate(rollout_data['info_history']):
            wingman_pos[t_idx,:] = [info['wingman']['y'], info['wingman']['x']]
            lead_pos[t_idx,:] = [info['lead']['y'], info['lead']['x']]
            rejoin_point_pos[t_idx, :] = [info['rejoin_region']['y'], info['rejoin_region']['x']]

            success_list.append(info['success'])
            failure_list.append(info['failure'])


        trajectory_data.append({'wingman_pos':wingman_pos, 'lead_pos':lead_pos, 'rejoin_point_pos':rejoin_point_pos, 'success':success_list, 'failure':failure_list})

    return trajectory_data

def main():

    expr_dir = 'output/expr_20201125_153608/'

    expr_data = pickle.load( open( os.path.join(expr_dir,"rollout_history.pickle"), "rb" ) )

    rollout_seq = expr_data['rollout_history']

    trajectory_data = process_rollout_data(rollout_seq)
    
    animate_trajectories(trajectory_data, os.path.join(expr_dir,'all_trajectories.mp4'))
    animate_trajectories([trajectory_data[-1]], os.path.join(expr_dir,'last_trajectory.mp4'), anim_rate=1, plot_rejoin_region=True, plot_safety_region=True, sq_axis=True)

if __name__ == '__main__':
    main()
