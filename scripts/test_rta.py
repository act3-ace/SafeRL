import math

from saferl import lookup
from saferl.environment.utils import YAMLParser

# import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib import animation

config = '../configs/rejoin/rejoin_default.yaml'

parser = YAMLParser(yaml_file=config, lookup=lookup)
env_class, env_config = parser.parse_env()

# timeout limit
env_config['status'][5]['config']['timeout'] = 30

# # wingman q1
# env_config['env_objs'][0]['config']['init'] = {
#     'x':707,
#     'y':-707,
#     'heading': 3*math.pi/4,
#     'v': 100,
# }

# # wingman q2
# env_config['env_objs'][0]['config']['init'] = {
#     'x':-707,
#     'y':-707,
#     'heading': math.pi/4,
#     'v': 100,
# }

# # wingman q3
# env_config['env_objs'][0]['config']['init'] = {
#     'x':-707,
#     'y':707,
#     'heading': -math.pi/4,
#     'v': 100,
# }

# wingman q4
env_config['env_objs'][0]['config']['init'] = {
    'x': 707,
    'y': 707,
    'heading': -3*math.pi/4,
    'v': 100,
}

# lead
env_config['env_objs'][1]['config']['init'] = {
    'x': -1000,
    'y': 0,
    'heading': 0,
    'v': 100,
}

env = env_class(config=env_config)

env.reset()
done = False

info_log = []

i = 0
while not done:
    obs, reward, done, info = env.step((None, None))

    info_log.append(info)

    print(i)
    i += 1

fig = plt.figure()
ax = plt.axes()
ax.set_aspect('equal', adjustable='box')
# ax.set(xlim=(-2000, 2000), ylim=(-1500, 100))
ax.set(xlim=(-2000, 2000), ylim=(-100, 1500))
ax.invert_yaxis()
wingman_marker, = plt.plot([], [], 'go')
lead_marker, = plt.plot([], [], 'bp')
wingman_traj, = plt.plot([], [], 'g')
lead_traj, = plt.plot([], [], 'b')


def init():
    wingman_marker.set_data([], [],)
    lead_marker.set_data([], [],)
    wingman_traj.set_data([], [],)
    lead_traj.set_data([], [],)


def animate(i):
    info = info_log[i]

    wingman_marker.set_data(info['wingman']['x'], info['wingman']['y'])
    lead_marker.set_data(info['lead']['x'], info['lead']['y'])
    wingman_traj.set_data(info['rta']['rta_traj'][:, 0], info['rta']['rta_traj'][:, 1])
    lead_traj.set_data(info['rta']['watch_traj'][:, 0], info['rta']['watch_traj'][:, 1])

    if info['rta']['rta_on']:
        wingman_traj.set_color('r')
    else:
        wingman_traj.set_color('g')

    return wingman_marker,


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(info_log), interval=50)

anim.save('basic_animation.mp4')
