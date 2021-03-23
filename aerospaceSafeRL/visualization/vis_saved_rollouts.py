import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.animation as animation
import os
import math
import matplotlib.patches as mpatches

import tqdm

from aerospaceSafeRL.AerospaceModels import CWHSpacecraft2d


def animate_trajectories_docking(rollout_seq, output_filename, colormap='jet', frame_interval=50, anim_rate=4, trail_length=40, plot_docking_region=False, plot_safety_region=False, color_type='g', sq_axis=False, extra_time=2, plot_estimated_trajectory=False, plot_actuators=False, actuator_config=None):
    ims = []

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(metadata=dict(artist='Umberto Ravaioli'), bitrate=1800)

    # compute axis dimensions
    all_xs = [ info['deputy']['x'] for rollout in rollout_seq for info in rollout['info_history'] ] + [ info['chief']['x'] for rollout in rollout_seq for info in rollout['info_history'] ]
    all_ys = [ info['deputy']['y'] for rollout in rollout_seq for info in rollout['info_history'] ] + [ info['chief']['y'] for rollout in rollout_seq for info in rollout['info_history'] ]

    max_x = max(all_xs)
    min_x = min(all_xs)
    max_y = max(all_ys)
    min_y = min(all_ys)

    fig = plt.figure()
    ax = plt.axes(xlim=(min_x-50, max_x+50), ylim=(min_y-50, max_y+50))

    if sq_axis:
        ax.set_aspect('equal', adjustable='box')

    max_time = max([len(rollout['info_history']) for rollout in rollout_seq])

    max_time = int(max_time + extra_time*1000/frame_interval*anim_rate)

    num_rollouts = len(rollout_seq)
    cmap = cm.get_cmap(colormap)

    # save location history of deputy for trails
    deputy_trails = [ ([], []) for i in range(num_rollouts) ]

    if plot_estimated_trajectory:
        trajectory_spacecraft = CWHSpacecraft2d()

    for t_idx in range(0, max_time, anim_rate):
        time_artists = []
        for rollout_idx, rollout in enumerate(rollout_seq):
            
            traj_len = len(rollout['info_history'])
            traj_time = min(t_idx, traj_len-1)
            info = rollout['info_history'][traj_time]

            deputy_pos = (info['deputy']['x'], info['deputy']['y'])
            chief_pos = (info['chief']['x'], info['chief']['y'])
            docking_region_params = (info['docking_region']['x'], info['docking_region']['y'], info['docking_region']['radius'])

            # save to trail
            deputy_trails[rollout_idx][0].append(deputy_pos[0])
            deputy_trails[rollout_idx][1].append(deputy_pos[1])

            # pop old trail entries
            if len(deputy_trails[rollout_idx][0]) > trail_length:
                deputy_trails[rollout_idx][0].pop(0)
                deputy_trails[rollout_idx][1].pop(0)

            fp_color = cmap(1 - rollout_idx/num_rollouts)

            if color_type == 'match':
                chief_color = fp_color
            else:
                chief_color = 'g'

            chief_marker = 'p'

            if info['success']:
                deputy_marker = '*'
            elif info['failure'] == 'failure_distance':
                deputy_marker = '4'
            elif info['failure'] == 'failure_timeout':
                deputy_marker = '+'
            elif info['failure'] == 'failure_crash':
                deputy_marker = 'x'
                chief_marker = 'x'
            else:
                deputy_marker = '^'

            if plot_estimated_trajectory:
                trajectory_spacecraft.state = info['deputy']['state']
                estimated_traj = trajectory_spacecraft.estimate_trajectory(time_window=200, num_points=100)

                estimated_traj_artist = plt.plot(estimated_traj[0,:], estimated_traj[1,:], ':', color=fp_color)

                time_artists += estimated_traj_artist          

            deputy_plot = plt.plot(deputy_pos[0], deputy_pos[1], color=fp_color, marker=deputy_marker)
            chief_plot = plt.plot(chief_pos[0], chief_pos[1], color=chief_color, marker=chief_marker)

            deputy_plot += plt.plot(deputy_trails[rollout_idx][0], deputy_trails[rollout_idx][1], color=fp_color)

            if plot_actuators:
                thrust_x = -1* info['deputy']['actuators']['thrust_x']
                thrust_y = -1 * info['deputy']['actuators']['thrust_y']
                
                thrust_x_norm_const = (actuator_config['thrust_x']['bounds'][1] - actuator_config['thrust_x']['bounds'][0])/2
                thrust_y_norm_const = (actuator_config['thrust_y']['bounds'][1] - actuator_config['thrust_y']['bounds'][0])/2

                thrust_x_norm = thrust_x / thrust_x_norm_const
                thrust_y_norm = thrust_y / thrust_y_norm_const

                thrust_scale = math.sqrt( (max_x - min_x) * (max_y - min_y) ) / 20
                thrust_angle = np.arctan2(thrust_y_norm, thrust_x_norm) * 180/math.pi
                thrust_mag = np.linalg.norm([thrust_x_norm, thrust_y_norm]) * thrust_scale

                thrust_artist = mpatches.Wedge(deputy_pos, thrust_mag, thrust_angle-10, thrust_angle+10, color='m')
                ax.add_patch(thrust_artist)
                deputy_plot.append(thrust_artist)

            if plot_docking_region or plot_safety_region:
                death_radius = 100
                docking_radius = 20

                docking_region_artists = []
                # docking_region_artists.append(plt.Circle((lead_pos[traj_time,0], lead_pos[traj_time,1]), docking_radius, color='g'))
                
                # docking_region_artists.append(plt.Circle((lead_pos[traj_time,0], lead_pos[traj_time,1]), rejoin_min_radius, color='w'))
                if plot_docking_region:
                    if color_type == 'match':
                        docking_color = fp_color
                    else:
                        docking_color = color_type
                    docking_region_artists.append(plt.Circle((docking_region_params[0], docking_region_params[1]), docking_region_params[2], color=docking_color, alpha=0.5))

                # if plot_safety_region:
                #     docking_region_artists.append(plt.Circle((lead_pos[traj_time,0], lead_pos[traj_time,1]), death_radius, color='r', alpha=0.5))


                for circle_artist in docking_region_artists:
                    ax.add_patch(circle_artist)

                time_artists += docking_region_artists


            time_artists += deputy_plot
            time_artists += chief_plot

        ims.append(time_artists)

    im_ani = animation.ArtistAnimation(fig, ims, interval=frame_interval, repeat_delay=3000,
                                   blit=True)
    im_ani.save(output_filename, writer=writer)

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
