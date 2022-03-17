import abc
import math
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
import argparse
from tqdm import tqdm

from scripts.eval import parse_jsonlines_log


def get_args():
    """
    A function to process script args.

    Returns
    -------
    argparse.Namespace
        Collection of command line arguments and their values
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('log', type=str, help="The full path to the trajectory log file")
    parser.add_argument('animator', type=str, help="docking_2d, docking_2d_oriented")
    parser.add_argument('--output', type=str, default="anim.mp4")

    return parser.parse_args()


class animator(abc.ABC):
    def __init__(self, log_data, concurrent=False):
        self.log_data = log_data
        self.concurrent = concurrent
        self.ui_scale_mat = np.eye(2)

        if self.concurrent:
            self.num_frames = max([len(d) for d in self.log_data])
        else:
            self.num_frames = sum([len(d) for d in self.log_data])

    def animate(self):
        self.fig, self.ax = plt.subplots(subplot_kw={'adjustable': 'box', 'aspect': 'equal'})
        self.ax.set_xlim(left=self.xlim[0], right=self.xlim[1])
        self.ax.set_ylim(bottom=self.ylim[0], top=self.ylim[1])

        xlim_span = self.xlim[1] - self.xlim[0]
        ylim_span = self.ylim[1] - self.ylim[0]

        self.ui_scale_mat = np.array([[max(xlim_span, ylim_span), 0], [0, max(xlim_span, ylim_span)]], dtype=float)

        self.setup_artists()

        anim = animation.FuncAnimation(
            self.fig, self.frame_change, frames=self.num_frames, init_func=self.frame_init, blit=True, interval=16.7)

        return anim

    def setup_artists(self):
        pass

    def get_sequential_index(self, frame):

        for i in range(len(self.log_data)):
            ep_len = len(self.log_data[i])
            if frame < ep_len:
                break
            else:
                frame -= ep_len

        return i, frame

    def frame_init(self):
        return []

    @abc.abstractmethod
    def frame_change(self, frame):
        raise NotImplementedError


class animator_docking_2d(animator):
    def __init__(self, log_data, concurrent=False):
        super().__init__(log_data, concurrent=concurrent)

        deputy_xs = [d['info']['deputy']['x'] for ep_log in self.log_data for d in ep_log]
        deputy_ys = [d['info']['deputy']['y'] for ep_log in self.log_data for d in ep_log]

        self.xlim = (min(deputy_xs+[0]), max(deputy_xs+[0]))
        self.ylim = (min(deputy_ys+[0]), max(deputy_ys+[0]))

        xlim_buffer = 0.1 * (self.xlim[1] - self.xlim[0])
        ylim_buffer = 0.1 * (self.ylim[1] - self.ylim[0])

        self.xlim = (self.xlim[0] - xlim_buffer, self.xlim[1] + xlim_buffer)
        self.ylim = (self.ylim[0] - ylim_buffer, self.ylim[1] + ylim_buffer)

    def setup_artists(self):
        self.deputy_patch = patches.Polygon(get_square_verts(0, 0, scale=0.1*self.ui_scale_mat).T)
        self.ax.add_patch(self.deputy_patch)

        self.deputy_thruster_x = patches.Polygon(thrust_wedge_verts.copy().T, fc='r')
        self.ax.add_patch(self.deputy_thruster_x)

        self.deputy_thruster_y = patches.Polygon(thrust_wedge_verts.copy().T, fc='r')
        self.ax.add_patch(self.deputy_thruster_y)

    def frame_change(self, frame):
         # return self._frame_change(self.log_data[frame])
        artists = []

        if not self.concurrent:
            ep_idx, ts_idx = self.get_sequential_index(frame)
            data = self.log_data[ep_idx][ts_idx]
        else:
            data = None

        x = data['info']['deputy']['x']
        y = data['info']['deputy']['y']
        control = data['info']['deputy']['controller']['control']
        thrust_x = control[0]
        thrust_y = control[1]

        # import pdb; pdb.set_trace()
        deputy_verts = get_square_verts(x, y, scale=0.1*self.ui_scale_mat)
        self.deputy_patch.set_xy(deputy_verts.T)
        artists.append(self.deputy_patch)

        thruster_verts = rotate_scale_transform_verts(thrust_wedge_verts.copy(), x, y, 0,
                                                      scale=0.2*thrust_x*self.ui_scale_mat)
        self.deputy_thruster_x.set_xy(thruster_verts.T)
        artists.append(self.deputy_thruster_x)

        thruster_verts = rotate_scale_transform_verts(thrust_wedge_verts.copy(), x, y, 0,
                                                      scale=0.2*thrust_y*self.ui_scale_mat)
        self.deputy_thruster_y.set_xy(thruster_verts.T)
        artists.append(self.deputy_thruster_y)

        return artists

class animator_docking_2d_oriented(animator):

    def __init__(self, log_data):
        super().__init__(log_data, concurrent=False)

        deputy_xs = [d['info']['deputy']['x'] for ep_log in self.log_data for d in ep_log]
        deputy_ys = [d['info']['deputy']['y'] for ep_log in self.log_data for d in ep_log]

        self.xlim = (min(deputy_xs+[0]), max(deputy_xs+[0]))
        self.ylim = (min(deputy_ys+[0]), max(deputy_ys+[0]))

        xlim_buffer = 0.1 * (self.xlim[1] - self.xlim[0])
        ylim_buffer = 0.1 * (self.ylim[1] - self.ylim[0])

        self.xlim = (self.xlim[0] - xlim_buffer, self.xlim[1] + xlim_buffer)
        self.ylim = (self.ylim[0] - ylim_buffer, self.ylim[1] + ylim_buffer)

    def animate(self):
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2, figsize=[6.4*2, 4.8])
        
        self.ax = self.axes[0]

        self.ax.set_aspect(adjustable='box', aspect='equal')

        self.ax.set_xlim(left=self.xlim[0], right=self.xlim[1])
        self.ax.set_ylim(bottom=self.ylim[0], top=self.ylim[1])

        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')

        xlim_span = self.xlim[1] - self.xlim[0]
        ylim_span = self.ylim[1] - self.ylim[0]

        self.ui_scale_mat = np.array([[max(xlim_span, ylim_span), 0], [0, max(xlim_span, ylim_span)]], dtype=float)

        self.setup_artists()

        anim = animation.FuncAnimation(
            self.fig, self.frame_change, frames=self.num_frames, init_func=self.frame_init, blit=True, interval=16.7)

        return anim

    def setup_artists(self):
        artists = []
        self.origin_patch = patches.Circle((0, 0), radius=10, fc='palegreen')
        self.ax.add_patch(self.origin_patch)
        artists.append(self.origin_patch)

        self.deputy_patch = patches.Polygon(get_chevron_verts(0, 0, scale=0.1*self.ui_scale_mat).T, fc='tab:blue')
        self.ax.add_patch(self.deputy_patch)
        artists.append(self.deputy_patch)

        self.deputy_thruster = patches.Polygon(thrust_wedge_verts.copy().T, fc='tab:red')
        self.ax.add_patch(self.deputy_thruster)
        artists.append(self.deputy_thruster)

        self.chief_patch = patches.Circle((0, 0), radius=0.5, fc='g')
        self.ax.add_patch(self.chief_patch)
        artists.append(self.chief_patch)

        self.draw_constraint_setup(self.axes[1])

        return artists

    def frame_change(self, frame):
        # return self._frame_change(self.log_data[frame])
        artists = []

        if not self.concurrent:
            ep_idx, ts_idx = self.get_sequential_index(frame)
            ep_data = self.log_data[ep_idx]
            data = ep_data[ts_idx]

            if ts_idx == 0:
                artists += self.draw_constraint(ep_data, self.axes[1])
        else:
            data = None

        x = data['info']['deputy']['x']
        y = data['info']['deputy']['y']
        theta = data['info']['deputy']['theta']
        thrust = data['info']['deputy']['controller']['control'][0]

        # import pdb; pdb.set_trace()
        deputy_verts = get_chevron_verts(x, y, theta, scale=0.1*self.ui_scale_mat)
        self.deputy_patch.set_xy(deputy_verts.T)
        artists.append(self.deputy_patch)

        thruster_verts = rotate_scale_transform_verts(thrust_wedge_verts.copy(), x, y, theta,
                                                      scale=0.2*thrust*self.ui_scale_mat)
        self.deputy_thruster.set_xy(thruster_verts.T)
        artists.append(self.deputy_thruster)

        artists += self.draw_constraint_current(data, self.axes[1])

        return artists

    def draw_constraint_setup(self, ax):
        self.dist_vel_artist, = ax.plot([], [], color='tab:blue')
        self.dist_vel_current_artist, = ax.plot([], [], color='tab:blue', marker='o')

        n = 0.001027
        v0 = 0.2
        constraint_dist = 1000
        constraint_val = 2*n*constraint_dist + v0
        self.constraint_limit_artist, = ax.plot([0, constraint_dist], [v0, constraint_val], 'k--')

        # compute max distance
        max_dist = 150
        for data in self.log_data:
            dist, _ = self.compute_dists_vels(data)
            max_dist = max(max_dist, np.max(dist))

        ax.set_xlim([0, max_dist])
        ax.set_ylim([0, 0.8])

        ax.set_xlabel('Relative Distance (m)')
        ax.set_ylabel('Relative Velocity (m/s)')
        ax.set_title('Distance Dependent Velocity Constraint')

        return [self.dist_vel_artist]

    def draw_constraint(self, ep_data, ax):
        dists, vels = self.compute_dists_vels(ep_data)
        self.dist_vel_artist.set_data(dists, vels)

        return [self.dist_vel_artist]

    def draw_constraint_current(self, data, ax):
        x = data['info']['deputy']['x']
        y = data['info']['deputy']['y']
        x_dot = data['info']['deputy']['x_dot']
        y_dot = data['info']['deputy']['y_dot']

        dist = math.sqrt(x**2 + y**2)
        vel = math.sqrt(x_dot**2 + y_dot**2)

        self.dist_vel_current_artist.set_data(dist, vel)

        return [self.dist_vel_current_artist]

    def compute_dists_vels(self, ep_data):
        x_s = np.array([data['info']['deputy']['x'] for data in ep_data], dtype=float)
        y_s = np.array([data['info']['deputy']['y'] for data in ep_data], dtype=float)

        x_dot_s = np.array([data['info']['deputy']['x_dot'] for data in ep_data], dtype=float)
        y_dot_s = np.array([data['info']['deputy']['y_dot'] for data in ep_data], dtype=float)

        dists = np.sqrt(x_s**2 + y_s**2)
        vels = np.sqrt(x_dot_s**2 + y_dot_s**2)

        return dists, vels

    def _frame_change(self, data):
        ...

    def ui_scale_polar(self, r, theta):
        x = r*np.cos(theta)
        y = r*np.sin(theta)

        x_scaled = self.ui_scale_mat[0, 0] * x
        y_scaled = self.ui_scale_mat[1, 1] * y

        r_scaled = math.sqrt(x_scaled**2 + y_scaled**2) * np.sign(r)
        theta_scaled = math.atan2(y_scaled, x_scaled)

        return r_scaled, theta_scaled


chevron_verts = np.array([
    [0, 0],
    [-0.2, 0.4],
    [0.6, 0],
    [-0.2, -0.4],
], dtype=float).T

thrust_wedge_verts = np.array([
    [0, 0],
    [-1, 0.26],
    [-1, -0.26],
]).T

square_verts = np.array([
    [-0.5, -0.5],
    [0.5, -0.5],
    [0.5, 0.5],
    [-0.5, 0.5],
]).T


def get_chevron_verts(x, y, theta=0, scale=1):
    verts = chevron_verts.copy()
    return rotate_scale_transform_verts(verts, x, y, theta, scale)


def get_square_verts(x, y, theta=0, scale=1):
    verts = square_verts.copy()
    return rotate_scale_transform_verts(verts, x, y, theta, scale)


def rotate_scale_transform_verts(verts, x, y, theta, scale=1):
    rot_mat = get_rot_mat_2d(theta)

    if not isinstance(scale, np.ndarray):
        scale = scale * np.eye(2)

    verts = (scale @ rot_mat @ verts) + np.array([[x, y]], dtype=float).T

    return verts


def get_rot_mat_2d(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ], dtype=float)


animator_map = {
    'docking_2d': animator_docking_2d,
    'docking_2d_oriented': animator_docking_2d_oriented,
}


def main():
    # process args
    args = get_args()
    log_data = parse_jsonlines_log(args.log, separate_episodes=True)

    # log_data = [data for data in log_data[0:3]]

    print(len(log_data))
    animator = animator_map[args.animator](log_data)

    anim = animator.animate()
    print("saving animation")

    save_bar = tqdm(total=animator.num_frames)
    anim.save(args.output, writer="ffmpeg", dpi=200,
              progress_callback=lambda i, n: save_bar.update(1))
    save_bar.close()
    # plt.show()


if __name__ == "__main__":
    main()
