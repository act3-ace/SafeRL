import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.animation as animation

import tqdm

def plot_trajectories(trajectory_data, output_filename, colormap='jet'):
    for traj in trajectory_data:
        wingman_pos = traj['wingman_pos']
        lead_pos = traj['lead_pos']

        plt.plot(wingman_pos[:,0],wingman_pos[:,1], color=fp_color)
        plt.plot(lead_pos[:,0],lead_pos[:,1], 'g')

    plt.savefig(output_filename)

def animate_trajectories(trajectory_data, output_filename, colormap='jet', anim_rate=4, trail_length=40):
    ims = []

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure()

    max_time = max([traj['wingman_pos'].shape[0] for traj in trajectory_data])

    num_rollouts = len(trajectory_data)
    cmap = cm.get_cmap(colormap)

    for t_idx in range(0, max_time, anim_rate):
        time_artists = []
        for traj_idx, traj in enumerate(trajectory_data):
            wingman_pos = traj['wingman_pos']
            lead_pos = traj['lead_pos']

            traj_len = wingman_pos.shape[0]
            traj_time = min(t_idx, traj_len-1)

            fp_color = cmap(1 - traj_idx/num_rollouts)

            trail_start = max(t_idx-trail_length, 0)
            wingman_trail = wingman_pos[trail_start:(traj_time+1),:]

            wingman_plot = plt.plot(wingman_pos[traj_time,0],wingman_pos[traj_time,1], color=fp_color, marker="o")
            lead_plot = plt.plot(lead_pos[traj_time,0],lead_pos[traj_time,1], 'g', marker="x")

            wingman_trail_plot = plt.plot(wingman_trail[:,0], wingman_trail[:,1], color=fp_color)


            time_artists += wingman_plot
            time_artists += lead_plot
            time_artists += wingman_trail_plot

        ims.append(time_artists)

    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=3000,
                                   blit=True)
    im_ani.save(output_filename, writer=writer)


def main():

    rollout_seq = pickle.load( open( "save.pickle", "rb" ) )

    num_rollouts = len(rollout_seq)

    cmap = cm.get_cmap('jet')

    trajectory_data = []

    for rollout_idx, rollout_data in tqdm.tqdm(enumerate(rollout_seq), total=len(rollout_seq)):
        wingman_pos = np.zeros((len(rollout_data['info_history']), 2))
        lead_pos = np.zeros((len(rollout_data['info_history']), 2))

        for t_idx, info in enumerate(rollout_data['info_history']):
            wingman_pos[t_idx,:] = [info['wingman']['y'], info['wingman']['x']]
            lead_pos[t_idx,:] = [info['lead']['y'], info['lead']['x']]

        fp_color = cmap(1 - rollout_idx/num_rollouts)

        trajectory_data.append({'wingman_pos':wingman_pos, 'lead_pos':lead_pos})

        plt.plot(wingman_pos[:,0],wingman_pos[:,1], color=fp_color)
        plt.plot(lead_pos[:,0],lead_pos[:,1], 'g')

    # plot_trajectories(trajectory_data, 'rollouts.png')

    animate_trajectories(trajectory_data, 'all_trajectories.mp4')
    animate_trajectories([trajectory_data[-1]], 'last_trajectory.mp4', anim_rate=1)

if __name__ == '__main__':
    main()