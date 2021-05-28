import math
import numpy as np

from saferl import lookup
from saferl.environment.utils import YAMLParser

# import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib import animation

config = '../configs/rejoin/rejoin_default.yaml'


def compute_collision_start(heading, v, collision_time=15):
    collision_dist = v * collision_time
    x = -1*collision_dist*math.cos(heading)
    y = -1*collision_dist*math.sin(heading)
    return x, y


def generate_collision_init():
    heading = np.random.uniform(-math.pi, math.pi)
    v = np.random.uniform(25, 100)
    x, y = compute_collision_start(heading, v)

    return {
        'x': x,
        'y': y,
        'heading': heading,
        'v': v,
    }


def generate_lead_wingman_collision():
    wingman_init = generate_collision_init()
    lead_init = generate_collision_init()

    while (wingman_init['x'] - lead_init['x'])**2 + (wingman_init['x'] - lead_init['x'])**2 < 300**2:
        lead_init = generate_collision_init()

    return wingman_init, lead_init


def run_collision(env_class, env_config, save_anim=False):
    wingman_init, lead_init = generate_lead_wingman_collision()
    env_config['env_objs'][0]['config']['init'] = wingman_init
    env_config['env_objs'][1]['config']['init'] = lead_init

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

    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes()
    ax.set_aspect('equal', adjustable='box')
    ax.set(xlim=(-2000, 2000), ylim=(-2000, 2000))
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

    plt.show()
    if save_anim:
        anim.save('basic_animation.mp4')


parser = YAMLParser(yaml_file=config, lookup=lookup)

np.random.seed(0)
for i in range(10):
    env_class, env_config = parser.parse_env()

    # timeout limit
    env_config['status'][5]['config']['timeout'] = 30
    run_collision(env_class, env_config)
