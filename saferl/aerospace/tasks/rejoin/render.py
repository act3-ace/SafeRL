"""
Rendering for 2D Dubins Rejoin Simulation

Created by Kai Delsing
Adapted by John McCarroll
Mentor: Kerianne Hobbs

Description:
    A class for rendering the 2D Dubins Rejoin environment.

renderSim:
    Create, run, and update the rendering
create_particle:
    Instantiate and initialize a particle object in the necessary lists
clean_particles:
    Delete particles past their ttl or all at once
close:
    Close the viewer and rendering
"""


import time
from gym.envs.classic_control import rendering


class RejoinRender:

    def __init__(self,
                 x_threshold=10000,
                 y_threshold=10000,
                 scale_factor=25,
                 show_rejoin=True,
                 show_ring=False,
                 plane_scale=1,
                 r_aircraft=15,
                 show_res=False,
                 termination_condition=False,
                 trace=1,
                 render_speed=0.03):

        self.x_threshold = x_threshold      # ft (To the left)
        self.y_threshold = y_threshold      # ft (Up or down)
        self.r_aircraft = r_aircraft        # ft - radius of the aircraft
        self.plane_scale = plane_scale      # dialation of aircraft size
        self.scale_factor = scale_factor
        self.render_speed = render_speed
        self.viewer = None

        # Toggle shown items
        self.show_ring = show_ring          # add ring around aircraft for visual aid
        self.show_rejoin = show_rejoin      # render rejoin region for visual aid
        self.showRes = show_res
        self.termination_condition = termination_condition  # Set to true to print termination condition

        # Trace params
        self.trace = trace                  # (steps) spacing between trace dots
        self.tracectr = self.trace

    def renderSim(self, state, mode='human'):
        # collect state data and set screen
        x_thresh = self.x_threshold / self.scale_factor
        y_thresh = self.y_threshold / self.scale_factor

        screen_width = x_thresh * 2
        screen_height = y_thresh * 2
        screen_width, screen_height = int(screen_width), int(screen_height)  # convert the screen width and height to integers

        wingwidth = 25 * self.r_aircraft * self.plane_scale / self.scale_factor
        wingheight = 5 * self.r_aircraft * self.plane_scale / self.scale_factor
        bodywidth = 5 * self.r_aircraft * self.plane_scale / self.scale_factor
        bodyheight = 20 * self.r_aircraft * self.plane_scale / self.scale_factor
        tailwidth = 10 * self.r_aircraft * self.plane_scale / self.scale_factor

        # process agent state
        wingman_state = state.env_objs["wingman"].state._vector

        # get position of wingman
        wingman_x = (wingman_state[0] + self.x_threshold) / self.scale_factor
        wingman_y = (wingman_state[1] + self.y_threshold) / self.scale_factor

        # process lead state
        lead_state = state.env_objs["lead"].state._vector

        # get position of lead
        lead_x = (lead_state[0] + self.x_threshold) / self.scale_factor
        lead_y = (lead_state[1] + self.y_threshold) / self.scale_factor

        # process rejoin state
        rejoin_region = state.env_objs["rejoin_region"]
        rejoin_region_shape = rejoin_region.shape

        rejoin_region_x = (rejoin_region_shape._center[0] + self.x_threshold) / self.scale_factor
        rejoin_region_y = (rejoin_region_shape._center[1] + self.y_threshold) / self.scale_factor
        rejoin_region_r = rejoin_region_shape.radius / self.scale_factor

        if self.showRes:
            print("Height: " + str(screen_height))
            print("Width: " + str(screen_width))
            self.showRes = False

        # draw animation render in viewer
        if self.viewer is None:
            # if no self.viewer exists, create it
            self.viewer = rendering.Viewer(screen_width, screen_height)  # creates a render

            b, t, l, r = 0, self.y_threshold * 2, 0, self.x_threshold * 2  # + self.x_goal  # creates body dimensions
            sky = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates body polygon
            # self.skytrans = rendering.Transform()  # allows body to be moved
            # sky.add_attr(self.skytrans)
            sky.set_color(1, 1, 1)  # sets color of body
            self.viewer.add_geom(sky)  # adds body to viewer

            # Create lead plane
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
            self.tailtrans = rendering.Transform(translation=(0, -bodyheight / 3))  # translates the tail to the end of the body
            tail.add_attr(self.tailtrans)
            tail.set_color(.3, .3, .5)  # sets color of tail
            self.viewer.add_geom(tail)  # adds tail to render

            # Create wingman plane
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
            self.tailtrans_target = rendering.Transform(translation=(0, -bodyheight / 3))  # translates the tail to the end of the body
            tail.add_attr(self.tailtrans_target)
            tail.set_color(0, 0, 0)  # sets color of tail
            self.viewer.add_geom(tail)  # adds tail to render

            if self.show_ring:
                ring = rendering.make_circle(wingwidth / 2, 30, False)  # creates ring dimensions
                self.ringtrans = rendering.Transform()  # allows ring to be moved
                ring.add_attr(self.ringtrans)
                ring.add_attr(self.bodytrans)  # sets ring as part of body
                ring.set_color(.9, .0, .0)  # sets color of ring
                self.viewer.add_geom(ring)  # adds ring into render

            if self.show_rejoin:
                ring = rendering.make_circle(rejoin_region_r, 30, False)  # creates ring dimensions
                self.rejoin_trans = rendering.Transform()  # allows ring to be moved
                ring.add_attr(self.rejoin_trans)
                ring.set_color(.9, .0, .0)  # sets color of ring
                self.rejoin_trans.set_translation(rejoin_region_x, rejoin_region_y)
                self.viewer.add_geom(ring)  # adds ring into render

        # render agent plane
        self.bodytrans.set_rotation(wingman_state[2])  # rotate body
        self.bodytrans.set_translation(wingman_x, wingman_y)  # translate body
        # self.tailtrans.set_rotation(wingman_state[2])  # rotate tail
        # self.tailtrans.set_translation(wingman_x - wingman_trans_x, wingman_y - wingman_trans_y)  # translate tail

        # render lead plane
        self.bodytrans_target.set_rotation(lead_state[2])  # rotate body
        self.bodytrans_target.set_translation(lead_x, lead_y)  # translate body
        # self.tailtrans_target.set_rotation(0)  # rotate tail
        # self.tailtrans_target.set_translation(lead_x - lead_trans_x, lead_y + lead_trans_y)  # translate tail

        # render rejoin region
        if self.show_rejoin:
            self.rejoin_trans.set_translation(rejoin_region_x, rejoin_region_y)

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
            self.tracetrans.set_translation(wingman_x, wingman_y)  # translate trace

        # sleep to slow down animation
        time.sleep(self.render_speed)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):  # if a viewer exists, close and kill it
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def reset(self):
        self.close()
