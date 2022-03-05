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

    return parser.parse_args()


class animator(abc.ABC):
    def __init__(self, log_data):
        self.log_data = log_data

    @abc.abstractmethod
    def animate(self):
        raise NotImplementedError

    @abc.abstractmethod
    def frame_init(self):
        raise NotImplementedError

    @abc.abstractmethod
    def frame_change(self, frame):
        raise NotImplementedError


class animator_docking_oriented_2d(animator):

    def __init__(self, log_data):
        super().__init__(log_data)

        deputy_xs = [d['info']['deputy']['x'] for d in self.log_data]
        deputy_ys = [d['info']['deputy']['y'] for d in self.log_data]

        self.xlim = (min(deputy_xs), max(deputy_xs))
        self.ylim = (min(deputy_ys), max(deputy_ys))

        xlim_buffer = 0.1 * (self.xlim[1] - self.xlim[0])
        ylim_buffer = 0.1 * (self.ylim[1] - self.ylim[0])

        self.xlim = (self.xlim[0] - xlim_buffer, self.xlim[1] + xlim_buffer)
        self.ylim = (self.ylim[0] - ylim_buffer, self.ylim[1] + ylim_buffer)

        self.ui_scale_mat = np.eye(2)

    def animate(self):
        self.fig, self.ax = plt.subplots(subplot_kw={'adjustable': 'box', 'aspect': 'equal'})
        self.ax.set_xlim(left=self.xlim[0], right=self.xlim[1])
        self.ax.set_ylim(bottom=self.ylim[0], top=self.ylim[1])

        xlim_span = self.xlim[1] - self.xlim[0]
        ylim_span = self.ylim[1] - self.ylim[0]

        self.ui_scale_mat = np.array([[max(xlim_span, ylim_span), 0], [0, max(xlim_span, ylim_span)]], dtype=float)

        anim = animation.FuncAnimation(
            self.fig, self.frame_change, frames=len(self.log_data), init_func=self.frame_init, blit=True, interval=16.7)

        return anim

    def frame_init(self):
        artists = []

        self.deputy_patch = patches.Polygon(get_chevron_verts(0, 0).T)
        self.ax.add_patch(self.deputy_patch)
        artists.append(self.deputy_patch)

        self.deputy_thruster = patches.Polygon(thrust_wedge_verts.copy().T, fc='r')
        self.ax.add_patch(self.deputy_thruster)
        artists.append(self.deputy_thruster)

        return artists

    def frame_change(self, frame):
        # return self._frame_change(self.log_data[frame])
        artists = []

        data = self.log_data[frame]
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

        return artists

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


def get_chevron_verts(x, y, theta=0, scale=1):
    verts = chevron_verts.copy()
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


def main():
    # process args
    args = get_args()
    log_data = parse_jsonlines_log(args.log)

    # log_data = log_data[0:100]

    print(len(log_data))
    animator = animator_docking_oriented_2d(log_data)

    anim = animator.animate()
    print("saving animation")

    save_bar = tqdm(total=len(log_data))
    anim.save('anim.mp4', writer="ffmpeg", dpi=200,
              progress_callback=lambda i, n: save_bar.update(1))
    save_bar.close()
    # plt.show()


if __name__ == "__main__":
    main()
