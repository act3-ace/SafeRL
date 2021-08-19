import math
from gym.envs.classic_control import rendering

from saferl.environment.tasks.env import BaseEnv


class DubinsRejoin(BaseEnv):

    def __init__(self, env_config):
        super().__init__(env_config)

    def reset(self):
        return super().reset()

    def _step_sim(self, action):
        self.sim_state.env_objs['lead'].step(self.step_size)
        self.sim_state.env_objs['wingman'].step(self.step_size, action)

    def generate_info(self):
        info = {
            'wingman': self.env_objs['wingman'].generate_info(),
            'lead': self.env_objs['lead'].generate_info(),
            'rejoin_region': self.env_objs['rejoin_region'].generate_info(),
            'failure': self.status['failure'],
            'success': self.status['success'],
            'status': self.status,
            'reward': self.reward_manager.generate_info(),
            'timestep_size': self.step_size
        }

        return info

    def render(self, mode='human'):
        # TODO:
        #  render rejoin region             [check]
        #  adjust screen width properly
        #  get correct values from state
        #  remove goal line                 [check?]

        x_thresh = self.x_threshold / self.scale_factor
        # goal = self.x_goal / self.scale_factor
        y_thresh = self.y_threshold / self.scale_factor

        screen_width = x_thresh + x_thresh  # calculate the screen width by adding the distance to the goal and the left threshold.
        # An extra x_threshold is added to provide buffer space
        screen_height = y_thresh * 2  # calculate the screen height by doubling the y thresh (up and down)
        screen_width, screen_height = int(screen_width), int(
            screen_height)  # convert the screen width and height to integers
        if self.showRes:
            print("Height: " + str(screen_height))
            print("Width: " + str(screen_width))
            self.showRes = False

        wingwidth = 25 * self.r_aircraft * self.planescale / self.scale_factor
        wingheight = 5 * self.r_aircraft * self.planescale / self.scale_factor
        bodywidth = 5 * self.r_aircraft * self.planescale / self.scale_factor
        bodyheight = 20 * self.r_aircraft * self.planescale / self.scale_factor
        tailwidth = 10 * self.r_aircraft * self.planescale / self.scale_factor

        if self.viewer is None:
            # if no self.viewer exists, create it
            self.viewer = rendering.Viewer(screen_width, screen_height)  # creates a render

            b, t, l, r = 0, self.y_threshold * 2, 0, self.x_threshold * 2 #+ self.x_goal  # creates body dimensions
            sky = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates body polygon
            self.skytrans = rendering.Transform()  # allows body to be moved
            sky.add_attr(self.skytrans)
            sky.set_color(.7, .7, .9)  # sets color of body
            self.viewer.add_geom(sky)  # adds body to viewer

            b, t, l, r = -bodywidth / 2, bodywidth / 2, bodyheight / 2, -bodyheight / 2  # creates body dimensions
            body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates body polygon
            self.bodytrans = rendering.Transform()  # allows body to be moved
            body.add_attr(self.bodytrans)
            body.set_color(.2, .2, .2)  # sets color of body
            self.viewer.add_geom(body)  # adds body to viewer

            b, t, l, r = -wingwidth / 2, wingwidth / 2, wingheight / 2, -wingheight / 2  # creates wing dimensions
            wing = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates wing polygon
            self.wingtrans = rendering.Transform()  # allows wing to be moved
            wing.add_attr(self.wingtrans)
            wing.add_attr(self.bodytrans)  # sets wing as part of body
            wing.set_color(.3, .5, .3)  # sets color of wing
            self.viewer.add_geom(wing)  # adds wing to viewer

            b, t, l, r = -tailwidth / 2, tailwidth / 2, wingheight / 2, -wingheight / 2  # creates tail dimensions
            tail = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates tail polygon
            self.tailtrans = rendering.Transform(
                translation=(0, -bodyheight / 3))  # translates the tail to the end of the body
            tail.add_attr(self.tailtrans)
            tail.set_color(.3, .3, .5)  # sets color of tail
            self.viewer.add_geom(tail)  # adds tail to render

            ########################################################################################################################
            b, t, l, r = -bodywidth / 2, bodywidth / 2, bodyheight / 2, -bodyheight / 2  # creates body dimensions
            body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates body polygon
            self.bodytrans_target = rendering.Transform()  # allows body to be moved
            body.add_attr(self.bodytrans_target)
            body.set_color(0, 0, 0)  # sets color of body
            self.viewer.add_geom(body)  # adds body to viewer

            b, t, l, r = -wingwidth / 2, wingwidth / 2, wingheight / 2, -wingheight / 2  # creates wing dimensions
            wing = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates wing polygon
            self.wingtrans_target = rendering.Transform()  # allows wing to be moved
            wing.add_attr(self.wingtrans_target)
            wing.add_attr(self.bodytrans_target)  # sets wing as part of body
            wing.set_color(0, 0, 0)  # sets color of wing
            self.viewer.add_geom(wing)  # adds wing to viewer

            b, t, l, r = -tailwidth / 2, tailwidth / 2, wingheight / 2, -wingheight / 2  # creates tail dimensions
            tail = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates tail polygon
            self.tailtrans_target = rendering.Transform(
                translation=(0, -bodyheight / 3))  # translates the tail to the end of the body
            tail.add_attr(self.tailtrans_target)
            tail.set_color(0, 0, 0)  # sets color of tail
            self.viewer.add_geom(tail)  # adds tail to render

            # goalLine = rendering.Line((goal, 2 * y_thresh), (goal, 0))  # creates goal line and endpoints
            # self.goalLinetrans = rendering.Transform()  # allows goalLine to be moved
            # goalLine.add_attr(self.goalLinetrans)
            # goalLine.set_color(.9, .1, .1)  # sets color of goalLine
            # self.viewer.add_geom(goalLine)  # adds goalLine into render

            # if self.ring:
            # TODO: create rejoin region ring?
            ring = rendering.make_circle(wingwidth / 2, 30, False)  # creates ring dimensions
            self.ringtrans = rendering.Transform()  # allows ring to be moved
            ring.add_attr(self.ringtrans)
            ring.add_attr(self.bodytrans)  # sets ring as part of body
            ring.set_color(.9, .0, .0)  # sets color of ring
            self.viewer.add_geom(ring)  # adds ring into render

        if self.state is None:  # if there is no state (either the simulation has not begun or it has ended), end
            print('No state')
            return None

        if self.trace != 0:
            if self.tracectr == self.trace:
                tracewidth = int(bodywidth / 2)
                if tracewidth < 1:
                    tracewidth = 1
                trace = rendering.make_circle(tracewidth)  # creates trace dot
                self.tracetrans = rendering.Transform()  # allows trace to be moved
                trace.add_attr(self.tracetrans)
                trace.set_color(.9, .1, .9)  # sets color of trace
                self.viewer.add_geom(trace)  # adds trace into render
                self.tracectr = 0
            else:
                self.tracectr += 1

        x = self.state
        tx, ty = x[0] / self.scale_factor, (
                    x[1] + self.y_threshold) / self.scale_factor  # pulls the state of the x and y coordinates
        self.bodytrans.set_rotation(x[2])  # rotate body
        self.bodytrans.set_translation(tx, ty)  # translate body

        self.tracetrans.set_translation(tx, ty)  # translate trace

        d = -bodyheight / 3  # set distance  #find distance to travel
        self.tailtrans.set_rotation(x[2])  # rotate tail
        thetashift = x[2] - 90.0  # convert graphics direction to Cartesian angles
        radtheta = (thetashift * 3.1415926535) / 180.0  # convert angle to radians
        transx, transy = math.sin(radtheta) * d, math.cos(radtheta) * d  # use trig to find actual x and y translations
        self.tailtrans.set_translation(tx - transx, ty + transy)  # translate tail

        ################################################################################################################################
        # TODO: basically duplicate code to render target plane
        tx, ty = self.x_target_plane / self.scale_factor, (
                    self.y_target_plane + self.y_threshold) / self.scale_factor  # get (x,y) of target
        # self.bodytrans_target.set_rotation(0)  #rotate body
        self.bodytrans_target.set_translation(tx, ty)  # translate body

        d = -bodyheight / 3  # set distance  #find distance to travel
        self.tailtrans_target.set_rotation(0)  # rotate tail
        thetashift = 0 - 90.0  # convert graphics direction to Cartesian angles
        radtheta = (thetashift * 3.1415926535) / 180.0  # convert angle to radians
        transx, transy = math.sin(radtheta) * d, math.cos(radtheta) * d  # use trig to find actual x and y translations
        self.tailtrans_target.set_translation(tx - transx, ty + transy)  # translate tail

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
