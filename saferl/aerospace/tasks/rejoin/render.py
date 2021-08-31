"""
Rendering for 2D Dubins Rejoin Simulation

Created by Kai Delsing
Adapted by John McCarroll
"""


import os
import time
from saferl.environment.tasks.render import BaseRenderer
if "DISPLAY" in os.environ.keys():
    from gym.envs.classic_control import rendering


class RejoinRenderer(BaseRenderer):

    def __init__(self,
                 x_threshold=10000,
                 y_threshold=10000,
                 scale_factor=25,
                 show_rejoin=True,
                 show_safety_ring=False,    # work in progress
                 safety_margin=None,
                 plane_scale=1,
                 r_aircraft=15,
                 show_res=False,
                 termination_condition=False,
                 trace=1,
                 render_speed=0.03):

        super().__init__()

        self.x_threshold = x_threshold      # ft (To the left)
        self.y_threshold = y_threshold      # ft (Up or down)
        self.r_aircraft = r_aircraft        # ft - radius of the aircraft
        self.plane_scale = plane_scale      # dialation of aircraft size
        self.scale_factor = scale_factor
        self.render_speed = render_speed
        self.safety_margin = safety_margin

        # Toggle shown items
        self.show_safety_ring = show_safety_ring          # add ring around aircraft for visual aid
        self.show_rejoin = show_rejoin      # render rejoin region for visual aid
        self.showRes = show_res
        self.termination_condition = termination_condition  # Set to true to print termination condition

        # Trace params
        self.trace = trace                  # (steps) spacing between trace dots
        self.tracectr = self.trace

    def render(self, state, mode='human'):
        # collect state data and set screen
        x_thresh = self.x_threshold / self.scale_factor
        y_thresh = self.y_threshold / self.scale_factor

        screen_width = x_thresh * 2
        screen_height = y_thresh * 2
        screen_width, screen_height = int(screen_width), int(screen_height)

        wingwidth = 15 * self.r_aircraft * self.plane_scale / self.scale_factor
        bodywidth = 5 * self.r_aircraft * self.plane_scale / self.scale_factor
        bodyheight = 20 * self.r_aircraft * self.plane_scale / self.scale_factor

        if not self.safety_margin:
            self.safety_margin = bodyheight / 2
        safety_ring_r = self.safety_margin / self.scale_factor

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
            sky.set_color(1, 1, 1)  # sets color of body
            self.viewer.add_geom(sky)  # adds body to viewer

            # creates body dimensions
            left_wing_x = -wingwidth / 2
            right_wing_x = wingwidth / 2
            wing_y = - bodyheight / 2
            nose_x = 0
            nose_y = bodyheight / 2

            # Create wingman plane
            wingman_body = rendering.FilledPolygon([(wing_y, left_wing_x), (wing_y, right_wing_x), (nose_y, nose_x)])
            self.wingman_transform = rendering.Transform()  # allows body to be moved
            wingman_body.add_attr(self.wingman_transform)
            wingman_body.set_color(1, 0, 1)  # sets color of body
            self.viewer.add_geom(wingman_body)  # adds body to viewer

            # Create lead plane
            lead_body = rendering.FilledPolygon([(wing_y, left_wing_x), (wing_y, right_wing_x), (nose_y, nose_x)])
            self.lead_transform = rendering.Transform()  # allows body to be moved
            lead_body.add_attr(self.lead_transform)
            lead_body.set_color(0, 0, 0)  # sets color of body
            self.viewer.add_geom(lead_body)  # adds body to viewer

            if self.show_safety_ring:
                # circle wingman
                ring = rendering.make_circle(safety_ring_r, 20, False)  # creates ring dimensions
                self.wingman_ring_transform = rendering.Transform()  # allows ring to be moved
                ring.add_attr(self.wingman_ring_transform)
                ring.add_attr(self.wingman_transform)  # sets ring as part of body
                ring.set_color(1, .0, .0)  # sets color of ring
                self.viewer.add_geom(ring)  # adds ring into render

                # circle lead too
                ring = rendering.make_circle(safety_ring_r, 20, False)  # creates ring dimensions
                self.lead_ring_transform = rendering.Transform()  # allows ring to be moved
                ring.add_attr(self.lead_ring_transform)
                ring.add_attr(self.lead_transform)  # sets ring as part of body
                ring.set_color(1, .0, .0)  # sets color of ring
                self.viewer.add_geom(ring)  # adds ring into render

            if self.show_rejoin:
                ring = rendering.make_circle(rejoin_region_r, 50, False)  # creates ring dimensions
                self.rejoin_trans = rendering.Transform()  # allows ring to be moved
                ring.add_attr(self.rejoin_trans)
                ring.set_color(1, .0, .0)  # sets color of ring
                self.rejoin_trans.set_translation(rejoin_region_x, rejoin_region_y)
                self.viewer.add_geom(ring)  # adds ring into render

        # render agent plane
        self.wingman_transform.set_rotation(wingman_state[2])  # rotate body
        self.wingman_transform.set_translation(wingman_x, wingman_y)  # translate body

        # render lead plane
        self.lead_transform.set_rotation(lead_state[2])  # rotate body
        self.lead_transform.set_translation(lead_x, lead_y)  # translate body

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
