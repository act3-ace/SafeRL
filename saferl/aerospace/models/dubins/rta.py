import math
import numpy as np

from saferl.environment.rta.rta import RTAModule
from saferl.environment.models.geometry import angle_wrap

class RTADubins2dCollision(RTAModule):

    def __init__(self):
        super().__init__()
        self.projection_window = 9 # seconds
        self.projection_frequency = 10
        self.watch_list = ['lead']
        self.platform_name = 'wingman'
        self.turn_rate = np.deg2rad(6)

        self.rta_on_dist = 200
        self.rta_off_dist = 400

        self.projection_numpoints = \
            self.projection_window * self.projection_frequency + 1

        self.rta_control = None
        self.rta_traj = None
        self.watch_traj = None

    def reset(self):
        super().reset()
        self.rta_control = None
        self.rta_traj = None
        self.watch_traj = None

    def _monitor(self, sim_state, control):
        rta_platform = sim_state.env_objs[self.platform_name]

        for watch_name in self.watch_list:

            watch_platform = sim_state.env_objs[watch_name]
            watch_traj = self.dubins_projection(watch_platform)

            rel_heading = angle_wrap(rta_platform.heading - watch_platform.heading)

            if (   ( rel_heading > 0          and rel_heading < math.pi/2)
                or ( rel_heading < -math.pi/2 and rel_heading > -math.pi ) ):
                # turn left
                rta_turn = -1 * self.turn_rate
            else:
                # turn right
                rta_turn = self.turn_rate

            if self.rta_on:
                rta_control_proposed = self.rta_control
            else:
                rta_control_proposed = np.array( [rta_turn, 0], dtype=np.float64 )

            rta_traj = self.dubins_projection(rta_platform, rta_control_proposed)

            # save trajectories
            self.rta_traj = rta_traj
            self.watch_traj = watch_traj

            traj_dist = np.linalg.norm( rta_traj -  watch_traj, axis=1)
            if (not self.rta_on) and (np.min(traj_dist) <= self.rta_on_dist):
                self.rta_on = True
                self.rta_control = rta_control_proposed
            elif self.rta_on and (np.min(traj_dist) > self.rta_off_dist):
                self.rta_on = False
                self.rta_control = None


    def _generate_control(self, sim_state, control):
        return np.copy(self.rta_control)

    def dubins_projection(self, platform, control=None):
        if control is None:
            control = np.copy(platform.current_control)

        base_traj = self.dubins_base_trajectory(platform.v, control)
        traj = platform.orientation.apply(base_traj) + platform.position[None, :]

        return traj[:, 0:2]
    
    def dubins_base_trajectory(self, v, control):
        turn_rate = control[0]

        if turn_rate == 0:
            traj = np.linspace(
                [0, 0, 0],
                [v*self.projection_window, 0, 0],
                self.projection_numpoints
            )
        else:
            end_theta = turn_rate * self.projection_window
            traj_theta = np.linspace(0, end_theta, self.projection_numpoints)
            turn_radius = v / turn_rate

            traj = np.zeros((self.projection_numpoints, 3))
            traj[:,0] = (turn_radius * np.cos(traj_theta - math.pi / 2))  
            traj[:,1] = (turn_radius * np.sin(traj_theta - math.pi / 2)) + turn_radius
            traj[:,2] = 0

        return traj

    def generate_info(self):
        return {
            'rta_on': self.rta_on,
            'rta_traj': self.rta_traj,
            'watch_traj': self.watch_traj
        }