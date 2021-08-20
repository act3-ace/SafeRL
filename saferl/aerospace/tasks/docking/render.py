'''
Rendering for 2D Spacecraft Docking Simulation

Created by Kai Delsing
Mentor: Kerianne Hobbs

Description:
    A class for rendering the SpacecraftDocking environment.

renderSim:
    Create, run, and update the rendering
create_particle:
    Instantiate and initialize a particle object in the necessary lists
clean_particles:
    Delete particles past their ttl or all at once
close:
    Close the viewer and rendering
'''


import math
import random
from gym.envs.classic_control import rendering


class DockingRender:

    def __init__(self, x_threshold=1500, y_threshold=1500, scale_factor=.25, show_res=False, velocity_arrow=False,
                 force_arrow=False, thrust_vis="Block", stars=500, termination_condition=False, ellipse_a1=200,
                 ellipse_a2=40, ellipse_b1=100, ellipse_b2=20, ellipse_quality=150, trace=5, trace_min=True):
        self.x_threshold = x_threshold  # 1.5 * deputy position
        self.y_threshold = y_threshold  # 1.5 * deputy position
        self.scale_factor = scale_factor # TODO: find out these magic numbers
        self.viewer = None

        self.bg_color = (0, 0, .15)  # r,g,b

        # Toggle shown items
        self.showRes = show_res
        self.velocityArrow = velocity_arrow
        self.forceArrow = force_arrow
        self.thrustVis = thrust_vis
        self.stars = stars
        self.termination_condition = termination_condition  # Set to true to print termination condition

        # Ellipse params
        self.ellipse_a1 = ellipse_a1  # m
        self.ellipse_b1 = ellipse_b1  # m
        self.ellipse_a2 = ellipse_a2  # m
        self.ellipse_b2 = ellipse_b2  # m
        self.ellipse_quality = ellipse_quality  # 1/x * pi

        # Trace params
        self.trace = trace  # (steps) spacing between trace dots
        self.traceMin = trace_min  # sets trace size to 1 (minimum) if true
        self.tracectr = self.trace

    def render(self, state, mode='human'):
        # create scale-adjusted variables
        x_thresh = self.x_threshold * self.scale_factor
        y_thresh = self.y_threshold * self.scale_factor
        screen_width, screen_height = int(x_thresh * 2), int(y_thresh * 2)

        if self.showRes:
            # print height and width
            print("Height: ", screen_height)
            print("Width: ", screen_width)
            self.showRes = False

        # create dimensions of satellites
        bodydim = 30 * self.scale_factor
        panelwid = 50 * self.scale_factor
        panelhei = 20 * self.scale_factor

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)  # create render viewer object

            # SKY #
            b, t, l, r = 0, y_thresh * 2, 0, x_thresh * 2  # creates sky dimensions
            sky = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates sky polygon
            self.skytrans = rendering.Transform()  # allows sky to be moved
            sky.add_attr(self.skytrans)
            sky.set_color(self.bg_color[0], self.bg_color[1], self.bg_color[2])  # sets color of sky
            self.viewer.add_geom(sky)  # adds sky to viewer

            # DEPUTY BODY #
            b, t, l, r = -bodydim, bodydim, -bodydim, bodydim
            deputy_body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates deputy body polygon
            self.deputy_bodytrans = rendering.Transform()  # allows body to be moved
            deputy_body.add_attr(self.deputy_bodytrans)
            deputy_body.set_color(.5, .5, .5)  # sets color of body

            # DEPUTY SOLAR PANEL #
            b, t, l, r = -panelhei, panelhei, -panelwid * 2, panelwid * 2
            deputy_panel = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates solar panel polygon
            self.deputy_panel_trans = rendering.Transform()  # allows panel to be moved
            deputy_panel.add_attr(self.deputy_panel_trans)  # sets panel as part of deputy object
            deputy_panel.add_attr(self.deputy_bodytrans)
            deputy_panel.set_color(.2, .2, .5)  # sets color of panel

            # CHIEF BODY #
            b, t, l, r = -bodydim, bodydim, -bodydim, bodydim
            chief_body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates body polygon
            self.chief_bodytrans = rendering.Transform()  # allows body to be moved
            chief_body.add_attr(self.chief_bodytrans)
            chief_body.set_color(.5, .5, .5)  # sets color of body

            # CHIEF SOLAR PANEL #
            b, t, l, r = -panelhei, panelhei, -panelwid * 2, panelwid * 2
            chief_panel = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates solar panel polygon
            self.chief_panel_trans = rendering.Transform()  # allows panel to be moved
            chief_panel.add_attr(self.chief_panel_trans)
            chief_panel.add_attr(self.chief_bodytrans)  # sets panel as part of chief object
            chief_panel.set_color(.2, .2, .5)  # sets color of panel

            # VELOCITY ARROW #
            if self.velocityArrow:
                velocityArrow = rendering.Line((0, 0), (panelwid * 2, 0))
                self.velocityArrowTrans = rendering.Transform()
                velocityArrow.add_attr(self.velocityArrowTrans)
                velocityArrow.add_attr(self.deputy_bodytrans)
                velocityArrow.set_color(.8, .1, .1)

            # FORCE ARROW #
            if self.forceArrow:
                forceArrow = rendering.Line((0, 0), (panelwid * 2, 0))
                self.forceArrowTrans = rendering.Transform()
                forceArrow.add_attr(self.forceArrowTrans)
                forceArrow.add_attr(self.deputy_bodytrans)
                forceArrow.set_color(.1, .8, .1)

            # THRUST BLOCKS #
            if self.thrustVis == 'Block':
                b, t, l, r = -bodydim / 2, bodydim / 2, -panelwid, panelwid # half the panel dimensions
                L_thrust = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates thrust polygon
                self.L_thrust_trans = rendering.Transform()  # allows thrust to be moved
                L_thrust.add_attr(self.L_thrust_trans)
                L_thrust.add_attr(self.deputy_bodytrans)
                L_thrust.set_color(.7, .3, .1)  # sets color of thrust
                R_thrust = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates thrust polygon
                self.R_thrust_trans = rendering.Transform()  # allows thrust to be moved
                R_thrust.add_attr(self.R_thrust_trans)
                R_thrust.add_attr(self.deputy_bodytrans)
                R_thrust.set_color(.7, .3, .1)  # sets color of thrust

                b, t, l, r = -bodydim / 2, bodydim / 2, -bodydim / 2, bodydim / 2
                T_thrust = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates thrust polygon
                self.T_thrust_trans = rendering.Transform()  # allows thrust to be moved
                T_thrust.add_attr(self.T_thrust_trans)
                T_thrust.add_attr(self.deputy_bodytrans)
                T_thrust.set_color(.7, .3, .1)  # sets color of thrust
                B_thrust = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates thrust polygon
                self.B_thrust_trans = rendering.Transform()  # allows thrust to be moved
                B_thrust.add_attr(self.B_thrust_trans)
                B_thrust.add_attr(self.deputy_bodytrans)
                B_thrust.set_color(.7, .3, .1)  # sets color of thrust

            # STARS #
            if self.stars > 0:
                for i in range(self.stars):
                    x, y = random.random() * (x_thresh * 2), random.random() * (y_thresh * 2)
                    dim = bodydim / 10
                    if dim <= 0:
                        dim = 1
                    star = rendering.make_circle(dim)  # creates trace dot
                    self.startrans = rendering.Transform()  # allows trace to be moved
                    star.add_attr(self.startrans)
                    star.set_color(.9, .9, .9)  # sets color of trace
                    self.viewer.add_geom(star)  # adds trace into render
                    self.startrans.set_translation(x, y)

            # ELLIPSES #
            if self.ellipse_quality > 0:
                thetaList = []
                i = 0
                while i <= math.pi * 2:
                    thetaList.append(i)
                    i += (1 / 100) * math.pi
                dotsize = int(self.scale_factor) + 1
                if dotsize < 0:
                    dotsize += 1

                for i in range(0, len(thetaList)):  # ellipse 1
                    x, y = self.ellipse_b1 * math.cos(thetaList[i]), self.ellipse_a1 * math.sin(thetaList[i])
                    x = (x * self.scale_factor) + x_thresh
                    y = (y * self.scale_factor) + y_thresh
                    dot1 = rendering.make_circle(dotsize)  # creates dot
                    self.dot1trans = rendering.Transform()  # allows dot to be moved
                    dot1.add_attr(self.dot1trans)
                    dot1.set_color(.1, .9, .1)  # sets color of dot
                    self.dot1trans.set_translation(x, y)
                    self.viewer.add_geom(dot1)  # adds dot into render

                for i in range(0, len(thetaList)): # ellipse 2
                    x, y = self.ellipse_b2 * math.cos(thetaList[i]), self.ellipse_a2 * math.sin(thetaList[i])
                    x = (x * self.scale_factor) + x_thresh
                    y = (y * self.scale_factor) + y_thresh
                    dot2 = rendering.make_circle(dotsize)  # creates dot
                    self.dot2trans = rendering.Transform()  # allows dot to be moved
                    dot2.add_attr(self.dot2trans)
                    dot2.set_color(.8, .9, .1)  # sets color of dot
                    self.dot2trans.set_translation(x, y)
                    self.viewer.add_geom(dot2)  # adds dot into render

            self.viewer.add_geom(chief_panel)  # adds solar panel to viewer
            self.viewer.add_geom(chief_body)  # adds satellites to viewer

            if self.thrustVis == 'Block':
                self.viewer.add_geom(L_thrust)  # adds solar panel to viewer
                self.viewer.add_geom(R_thrust)  # adds solar panel to viewer
                self.viewer.add_geom(T_thrust)  # adds solar panel to viewer
                self.viewer.add_geom(B_thrust)  # adds thrust into viewer

            self.viewer.add_geom(deputy_panel)  # adds solar panel to viewer

            if self.velocityArrow:
                self.viewer.add_geom(velocityArrow)  # adds velocityArrow to viewer

            if self.forceArrow:
                self.viewer.add_geom(forceArrow)  # adds forceArrow to viewer

            self.viewer.add_geom(deputy_body)  # adds body to viewer

        if state is None:  # if there is no state (either the simulation has not begun or it has ended), end
            print('No state')
            return None

        deputy = state.env_objs["deputy"]
        chief = state.env_objs["chief"]

        tx, ty = (deputy.position[0] + self.x_threshold) * self.scale_factor, (deputy.position[1] + self.y_threshold) * self.scale_factor  # pulls the state of the x and y coordinates
        self.deputy_bodytrans.set_translation(tx, ty)  # translate deputy
        self.chief_bodytrans.set_translation(chief.position[0] + x_thresh, chief.position[1] + y_thresh)  # translate chief

        # PARTICLE DYNAMICS #
        if self.thrustVis == 'Particle':
            lx, ly = (x[0]) * self.scale_factor, (x[1]) * self.scale_factor
            v = random.randint(-self.p_var, self.p_var)
            if self.x_force > 0:
                DockingRender.create_particle(self, self.p_velocity, 180 + v, lx, ly, self.p_ttl)
            elif self.x_force < 0:
                DockingRender.create_particle(self, self.p_velocity, 0 + v, lx, ly, self.p_ttl)
            if self.y_force > 0:
                DockingRender.create_particle(self, self.p_velocity, 270 + v, lx, ly, self.p_ttl)
            elif self.y_force < 0:
                DockingRender.create_particle(self, self.p_velocity, 90 + v, lx, ly, self.p_ttl)

            for i in range(0, len(self.particles)):
                # velocity, theta, x, y, ttl
                self.particles[i][4] -= 1 # decrement the ttl
                r = (self.particles[i][1] * math.pi) / 180
                self.particles[i][2] += (self.particles[i][0] * math.cos(r))
                self.particles[i][3] += (self.particles[i][0] * math.sin(r))

            DockingRender.clean_particles(self, False)

            # translate & rotate all particles
            for i in range(0, len(self.p_obj)):
                self.trans[i].set_translation(x_thresh + self.particles[i][2], y_thresh + self.particles[i][3])  #translate particle
                self.trans[i].set_rotation(self.particles[i][1])

        # TRACE DOTS #
        if self.trace != 0:  # if trace enabled, draw trace
            if self.tracectr == self.trace:  # if time to draw a trace, draw, else increment counter
                if self.traceMin:
                    tracewidth = 1
                else:
                    tracewidth = int(bodydim / 8) + 1

                trace = rendering.make_circle(tracewidth)  # creates trace dot
                self.tracetrans = rendering.Transform()  # allows trace to be moved
                trace.add_attr(self.tracetrans)
                trace.set_color(.9, .1, .9)  # sets color of trace
                self.viewer.add_geom(trace)  # adds trace into render
                self.tracectr = 0
            else:
                self.tracectr += 1

        self.tracetrans.set_translation(tx, ty)  # translate trace

        # BLOCK THRUSTERS #
        if self.thrustVis == 'Block':
            inc_l, inc_r, inc_b, inc_t = -25, 25, -5, 5  # create block dimensions
            # calculate block translations
            if deputy.x_dot > 0:
                inc_l = -65 * self.scale_factor
                inc_r = 25 * self.scale_factor
            elif deputy.x_dot < 0:
                inc_r = 65 * self.scale_factor
                inc_l = -25 * self.scale_factor
            if deputy.y_dot > 0:
                inc_b = -35 * self.scale_factor
                inc_t = 5 * self.scale_factor
            elif deputy.y_dot < 0:
                inc_t = 35 * self.scale_factor
                inc_b = -5 * self.scale_factor

            # translate blocks
            self.L_thrust_trans.set_translation(inc_l, 0)
            self.R_thrust_trans.set_translation(inc_r, 0)
            self.T_thrust_trans.set_translation(0, inc_t)
            self.B_thrust_trans.set_translation(0, inc_b)

        # VELOCITY ARROW #
        if self.velocityArrow:
            tv = math.atan(deputy.y_dot / deputy.x_dot)  # angle of velocity
            if deputy.x_dot < 0: # arctan adjustment
                tv += math.pi
            self.velocityArrowTrans.set_rotation(tv)

        # FORCE ARROW #
        if self.forceArrow:
            if self.x_force == 0:
                tf = math.atan(0) #angle of velocity
            else:
                tf = math.atan(self.y_force / self.x_force)  # angle of velocity
            if self.x_force < 0:  # arctan adjustment
                tf += math.pi
            self.forceArrowTrans.set_rotation(tf)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def create_particle(self, velocity, theta, x, y, ttl):
        p = [velocity, theta, x, y, ttl]
        obj_len = len(self.p_obj)  # position of particle in list
        p_len = len(self.particles)  # position of particle in list
        trans_len = len(self.trans)  # position of particle in list

        self.particles.append(p)
        self.p_obj.append(self.particles[p_len])
        self.p_obj[obj_len] = rendering.make_circle(1)  # creates particle dot
        self.trans.append(rendering.Transform())  # allows particle to be moved
        self.p_obj[obj_len].add_attr(self.trans[trans_len])
        self.p_obj[obj_len].set_color(.9, .9, .6)  # sets color of particle

        self.trans[trans_len].set_translation(self.particles[p_len][2], self.particles[p_len][3])  # translate particle
        self.trans[trans_len].set_rotation(self.particles[p_len][1])
        self.viewer.add_geom(self.p_obj[obj_len])  # adds particle into render

        DockingRender.clean_particles(self, False)
        return p

    def clean_particles(self, all):
        while self.particles and (all or self.particles[0][4] < 0):  # if all or if the first particle has reached its ttl
            self.p_obj[0].set_color(self.bg_color[0], self.bg_color[1], self.bg_color[2])  # sets color of particle
            self.particles.pop(0)  # delete particle at beginning of list
            self.p_obj.pop(0)  # position of particle in list
            self.trans.pop(0)  # position of particle in list

    def close(self):  # if a viewer exists, close and kill it
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
