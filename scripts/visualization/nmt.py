import math
import matplotlib.pyplot as plt
import numpy as np
from saferl.aerospace.models.cwhspacecraft.platforms.cwh import CWHSpacecraft2d
import tqdm

cwh_spacecraft = CWHSpacecraft2d('deputy', integration_method='RK45')


n = 0.001027
y_init = 100
x_init = 50
t_max = 10000
ts = 1


v0_x_a_list = [
    0,
    0.5,
    1,
    2,
    5,
]

v0_x_list = [-1*a*n*x_init for a in v0_x_a_list]

v0_y_a_list = [
    0.1,
    0.2,
    0.5,
    1,
    2,
]

v0_y_list = [a*n*y_init for a in v0_y_a_list]

x_init_states = [[x_init, 0, 0, v] for v in v0_x_list]

y_init_states = [[0, y_init, v, 0] for v in v0_y_list]

v_init = 0.5*n*100
theta_list = [
    0,
    45,
    90,
    135,
    # 180,
    -135,
    -90,
    -45,
]

theta_init_states = [[0, y_init, v_init*math.cos(math.radians(theta)), v_init*math.sin(math.radians(theta))] for theta in theta_list]


def nmt_traj(init_states, t_max_input):
    trajectories = []
    for x0 in tqdm.tqdm(init_states):
        t_max = t_max_input

        x, y, x_dot, y_dot = x0
        cwh_spacecraft.reset(x=x, y=y, x_dot=x_dot, y_dot=y_dot)

        trajectory = np.zeros((math.ceil(t_max/ts)+1, 4))

        for i in range(math.ceil(t_max/ts)+1):
            state = cwh_spacecraft.state.vector
            trajectory[i, :] = state
            cwh_spacecraft.step(None, ts)

        trajectories.append(trajectory)

    return trajectories


# trajectories = nmt_traj(x_init_states, t_max)

# fig, ax = plt.subplots()
# ax.set_xlim([-150, 150])
# ax.set_ylim([-150, 150])
# for trajectory in trajectories:
#     ax.plot(trajectory[:, 0], trajectory[:, 1])

# plt.savefig('figs/nmt_x_start.png', dpi=300)

trajectories = nmt_traj(y_init_states, t_max)

fig, ax = plt.subplots()
ax.set_xlim([-150, 150])
ax.set_ylim([-150, 150])
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.plot(0, 0, "g+")
ax.set_title("NMT Varied x Velocity")
for trajectory in trajectories:
    ax.plot(trajectory[:, 0], trajectory[:, 1])

plt.savefig('figs/nmt_y_start.png', dpi=300)


trajectories = nmt_traj(theta_init_states, t_max)

fig, ax = plt.subplots()
ax.set_xlim([-150, 150])
ax.set_ylim([-150, 150])
ax.set_title("NMT Varied Velocity Angle")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.plot(0, 0, "g+")
for trajectory in trajectories:
    ax.plot(trajectory[:, 0], trajectory[:, 1])

plt.savefig('figs/nmt_theta_start.png', dpi=300)