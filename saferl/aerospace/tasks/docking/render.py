import os
import math
import random
from saferl.environment.tasks.render import BaseRenderer
if "DISPLAY" in os.environ.keys():
    from gym.envs.classic_control import rendering


class DockingRenderer(BaseRenderer):
    """
    Rendering for 2D Spacecraft Docking Simulation

    Created by Kai Delsing

    Description:
        A class for rendering the 2D Docking environment.

    render:
        Create, run, and update the rendering
    reset:
        Set the rendering and viewer to an initial state
    close:
        Close the viewer and rendering
    """

    def __init__(self, max_distance=150, padding=50, velocity_arrow=False, dot_size=1,
                 force_arrow=False, thrust_vis="Block", stars=500, ellipse_a=200,
                 ellipse_b=100, draw_ellipse=True, trace=5):
        super().__init__()
        self.screen_width = 750
        self.screen_height = 750
        self.scale_factor = ((self.screen_width - padding) // 2) / max_distance
        self.x_thresh = self.screen_width // 2
        self.y_thresh = self.screen_height // 2

        self.bg_color = (0, 0, .15)  # r,g,b

        # Toggle shown items
        self.show_vel_arrow = velocity_arrow
        self.show_force_arrow = force_arrow
        self.thrust_vis = thrust_vis
        self.stars = stars

        # Ellipse params
        self.ellipse_a = ellipse_a  # m
        self.ellipse_b = ellipse_b  # m
        self.draw_ellipse = draw_ellipse  # 1/x * pi

        # Trace params
        self.trace = trace  # (steps) spacing between trace dots
        self.dot_size = dot_size
        self.tracectr = self.trace

        # Dynamic Objects
        self.sky = None
        self.chief = None
        self.deputy = None
        self.docking_region = None
        self.particle_system = None
        self.velocity_arrow = None
        self.force_arrow = None

    def make_sky(self, color):
        # SKY #
        b, t, l, r = 0, self.y_thresh * 2, 0, self.x_thresh * 2  # creates sky dimensions
        sky = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates sky polygon
        sky_trans = rendering.Transform()  # allows sky to be moved
        sky.add_attr(sky_trans)
        sky.set_color(color[0], color[1], color[2])  # sets color of sky
        return sky, sky_trans

    def make_satellite(self, bodydim, panel_width, panel_height):
        # BODY #
        b, t, l, r = -bodydim, bodydim, -bodydim, bodydim
        body = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates body polygon
        body_trans = rendering.Transform()  # allows body to be moved
        body.add_attr(body_trans)
        body.set_color(.5, .5, .5)  # sets color of body

        # SOLAR PANEL #
        b, t, l, r = -panel_height, panel_height, -panel_width * 2, panel_width * 2
        panel = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])  # creates solar panel polygon
        panel_trans = rendering.Transform()  # allows panel to be moved
        panel.add_attr(panel_trans)
        panel.add_attr(body_trans)  # sets panel as part of deputy object
        panel.set_color(.2, .2, .5)  # sets color of panel

        return body, body_trans, panel, panel_trans

    def make_region(self, radius, parent_trans, color):
        region = rendering.make_circle(radius=radius, filled=False)
        region_trans = rendering.Transform()
        region.add_attr(region_trans)
        region.add_attr(parent_trans)
        region.set_color(color[0], color[1], color[2])
        return region, region_trans

    def make_arrow(self, parent_trans, color, start, end):
        arrow = rendering.Line(start, end)
        arrow_trans = rendering.Transform()
        arrow.add_attr(arrow_trans)
        arrow.add_attr(parent_trans)
        arrow.set_color(color[0], color[1], color[2])
        return arrow, arrow_trans

    def make_dot(self, size, color, position):
        x, y = position
        dot = rendering.make_circle(size)  # creates dot
        dot_trans = rendering.Transform()  # allows dot to be moved
        dot.add_attr(dot_trans)
        dot.set_color(color[0], color[1], color[2])  # sets color of dot
        dot_trans.set_translation(x, y)
        return dot, dot_trans

    def make_stars(self, num_stars, dim, color):
        stars = []
        for i in range(num_stars):
            x, y = random.random() * (self.x_thresh * 2), random.random() * (self.y_thresh * 2)
            if dim <= 0:
                dim = 1
            stars.append(self.make_dot(size=dim, color=color, position=(x, y)))
        return stars

    def make_ellipse(self, a, b, dot_size, color):
        theta_list = []
        i = 0
        while i <= math.pi * 2:
            theta_list.append(i)
            i += (1 / 500) * math.pi

        dots = []
        for i in range(0, len(theta_list)):  # ellipse 1
            x, y = b * math.cos(theta_list[i]), a * math.sin(theta_list[i])
            x = (x * self.scale_factor) + self.x_thresh
            y = (y * self.scale_factor) + self.y_thresh
            dots.append(self.make_dot(size=dot_size, color=color, position=(x, y)))
        return dots

    def initial_view(self, state):
        # screen_width, screen_height = int(self.x_thresh * 2), int(self.y_thresh * 2)

        # create dimensions of satellites
        bodydim = 8
        panelwid = 14
        panelhei = 6

        self.viewer = rendering.Viewer(self.screen_width, self.screen_height)  # create render viewer object

        # SKY #
        self.sky = self.make_sky(color=self.bg_color)
        self.viewer.add_geom(self.sky[0])  # adds sky to viewer

        # CHIEF AND DEPUTY #
        self.deputy = self.make_satellite(
            bodydim=bodydim,
            panel_width=panelwid,
            panel_height=panelhei
        )
        self.chief = self.make_satellite(
            bodydim=bodydim,
            panel_width=panelwid,
            panel_height=panelhei
        )

        # DOCKING REGION #
        docking_radius = state.env_objs["docking_region"].radius * self.scale_factor
        self.docking_region = self.make_region(
            radius=docking_radius,
            parent_trans=self.chief[1],
            color=(.9, 0, 0)
        )

        # STARS #
        if self.stars > 0:
            stars = self.make_stars(
                num_stars=self.stars,
                dim=self.dot_size,
                color=(.9, .9, .9)
            )
            for star, star_trans in stars:
                self.viewer.add_geom(star)  # adds trace into render

        # ELLIPSES #
        if self.draw_ellipse:
            ellipse_1 = self.make_ellipse(
                a=self.ellipse_a,
                b=self.ellipse_b,
                color=(.1, .9, .1),
                dot_size=self.dot_size
            )
            for dot, dot_trans in ellipse_1:
                self.viewer.add_geom(dot)

        self.viewer.add_geom(self.chief[2])  # adds chief solar panel to viewer
        self.viewer.add_geom(self.chief[0])  # adds chief body to viewer

        self.viewer.add_geom(self.docking_region[0])  # adds docking region to viewer

        # THRUST BLOCKS #
        if self.thrust_vis == 'Block':
            self.particle_system = ThrustBlocks(
                viewer=self.viewer,
                parent_trans=self.deputy[1],
                scale_factor=self.scale_factor,
                bodydim=bodydim,
                panelwid=panelwid,
                x_thresh=self.x_thresh,
                y_thresh=self.y_thresh
            )
        elif self.thrust_vis == 'Particles':
            self.particle_system = ThrustParticles(
                viewer=self.viewer,
                bg_color=self.bg_color,
                x_thresh=self.x_thresh,
                y_thresh=self.y_thresh,
                scale_factor=self.scale_factor
            )

        self.viewer.add_geom(self.deputy[2])  # adds deputy solar panel to viewer

        # VELOCITY ARROW #
        if self.show_vel_arrow:
            self.velocity_arrow = self.make_arrow(
                parent_trans=self.deputy[1],
                color=(.8, .1, .1),
                start=(0, 0),
                end=(panelwid * 2, 0)
            )
            self.viewer.add_geom(self.velocity_arrow[0])  # adds velocity arrow to viewer

        # FORCE ARROW #
        if self.show_force_arrow:
            self.force_arrow = self.make_arrow(
                parent_trans=self.deputy[1],
                color=(.1, .8, .1),
                start=(0, 0),
                end=(panelwid * 2, 0)
            )
            self.viewer.add_geom(self.force_arrow[0])  # adds velocity arrow to viewer

        self.viewer.add_geom(self.deputy[0])  # adds deputy body to viewer

    def render(self, state, mode='human'):
        if self.viewer is None:
            self.initial_view(state=state)

        # Get current state
        deputy_state = state.env_objs["deputy"]
        chief_state = state.env_objs["chief"]
        x_force, y_force = deputy_state.current_control

        # Pull and scale deputy's position
        tx = (deputy_state.position[0] + self.x_thresh / self.scale_factor) * self.scale_factor
        ty = (deputy_state.position[1] + self.y_thresh / self.scale_factor) * self.scale_factor

        # Translate satellite geometry
        self.deputy[1].set_translation(tx, ty)
        self.chief[1].set_translation(
            chief_state.position[0] + self.x_thresh, chief_state.position[1] + self.y_thresh)

        # Update particle system
        if self.particle_system is not None:
            self.particle_system.update(state=state)

        # Add trace
        if self.tracectr % self.trace == 0:  # if time to draw a trace, draw, else increment counter
            trace, trace_trans = self.make_dot(size=self.dot_size, color=(.9, .1, .9), position=(tx, ty))
            self.viewer.add_geom(trace)  # adds trace into render
        self.tracectr += 1

        # VELOCITY ARROW #
        if self.velocity_arrow is not None:
            tv = math.atan(deputy_state.y_dot / deputy_state.x_dot)  # angle of velocity
            if deputy_state.x_dot < 0:  # arctan adjustment
                tv += math.pi
            self.velocity_arrow[1].set_rotation(tv)

        # FORCE ARROW #
        if self.force_arrow is not None:
            if x_force == 0:
                tf = math.atan(0)  # angle of velocity
            else:
                tf = math.atan(y_force / x_force)  # angle of velocity
            if x_force < 0:  # arctan adjustment
                tf += math.pi
            self.force_arrow[1].set_rotation(tf)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


class ParticleSystem:
    def __init__(self, viewer, x_thresh, y_thresh, scale_factor):
        self.x_thresh = x_thresh
        self.y_thresh = y_thresh
        self.scale_factor = scale_factor
        self.viewer = viewer

    def update(self, state):
        raise NotImplementedError


class ThrustBlocks(ParticleSystem):
    def __init__(self, viewer, parent_trans, bodydim, panelwid, scale_factor, x_thresh, y_thresh):
        super().__init__(viewer=viewer, x_thresh=x_thresh, y_thresh=y_thresh, scale_factor=scale_factor)
        self.scale_factor = scale_factor
        self.bodydim = bodydim
        self.panelwid = panelwid
        self.l_thrust = self.make_thrust_block(
            coords=[-bodydim / 2, bodydim / 2, -panelwid, panelwid],
            parent_trans=parent_trans,
            color=(.7, .3, .1)
        )
        self.r_thrust = self.make_thrust_block(
            coords=[-bodydim / 2, bodydim / 2, -panelwid, panelwid],
            parent_trans=parent_trans,
            color=(.7, .3, .1)
        )
        self.t_thrust = self.make_thrust_block(
            coords=[-bodydim / 2, bodydim / 2, -bodydim / 2, bodydim / 2],
            parent_trans=parent_trans,
            color=(.7, .3, .1)
        )
        self.b_thrust = self.make_thrust_block(
            coords=[-bodydim / 2, bodydim / 2, -bodydim / 2, bodydim / 2],
            parent_trans=parent_trans,
            color=(.7, .3, .1)
        )
        self.viewer.add_geom(self.l_thrust[0])  # adds left thrust block to viewer
        self.viewer.add_geom(self.r_thrust[0])  # adds right thrust block to viewer
        self.viewer.add_geom(self.t_thrust[0])  # adds top thrust block to viewer
        self.viewer.add_geom(self.b_thrust[0])  # adds bottom thrust block to viewer

    def update(self, state):
        deputy_state = state.env_objs["deputy"]
        x_force, y_force = deputy_state.current_control
        lr_size = (-self.bodydim / 2, self.bodydim / 2, -self.panelwid, self.panelwid)
        tb_size = (-self.bodydim / 2, self.bodydim / 2, -self.bodydim / 2, self.bodydim / 2)
        inc_l, inc_r, inc_b, inc_t = 0, 0, 0, 0  # create block dimensions
        # calculate block translations
        if x_force > 0:
            # Hide right, show left
            inc_l = lr_size[2] * 2
            inc_r = lr_size[2] * 2
        elif x_force < 0:
            inc_l = lr_size[3] * 2
            inc_r = lr_size[3] * 2
        if y_force > 0:
            # Hide top, show bottom
            inc_b = tb_size[0] * 2
            inc_t = tb_size[0] * 2
        elif y_force < 0:
            inc_b = tb_size[1] * 2
            inc_t = tb_size[1] * 2

        # translate blocks
        self.l_thrust[1].set_translation(inc_l, 0)
        self.r_thrust[1].set_translation(inc_r, 0)
        self.t_thrust[1].set_translation(0, inc_t)
        self.b_thrust[1].set_translation(0, inc_b)

    @staticmethod
    def make_thrust_block(coords, parent_trans, color):
        bottom, top, left, right = coords
        thrust = rendering.FilledPolygon(
            [(left, bottom), (left, top), (right, top), (right, bottom)])  # creates thrust polygon
        thrust_trans = rendering.Transform()  # allows thrust to be moved
        thrust.add_attr(thrust_trans)
        thrust.add_attr(parent_trans)
        thrust.set_color(color[0], color[1], color[2])  # sets color of thrust
        return thrust, thrust_trans


class ThrustParticles(ParticleSystem):
    def __init__(self, viewer, bg_color, x_thresh, y_thresh, scale_factor):
        # Thrust & Particle Variables #
        super().__init__(viewer=viewer, x_thresh=x_thresh, y_thresh=y_thresh, scale_factor=scale_factor)
        self.bg_color = bg_color
        self.particles = []  # list containing particle references
        self.p_obj = []  # list containing particle objects
        self.trans = []  # list containing particle
        self.p_velocity = 20  # velocity of particle
        self.p_ttl = 4  # (steps) time to live per particle
        self.p_var = 3  # (deg) the variation of launch angle (multiply by 2 to get full angle)

    def update(self, state):
        deputy_state = state.env_objs["deputy"]
        x, y = (deputy_state.position[0]) * self.scale_factor, (deputy_state.position[1]) * self.scale_factor
        x_force, y_force = deputy_state.current_control
        v = random.randint(-self.p_var, self.p_var)
        if x_force > 0:
            self.create_particle(180 + v, x, y)
        elif x_force < 0:
            self.create_particle(0 + v, x, y)
        if y_force > 0:
            self.create_particle(270 + v, x, y)
        elif y_force < 0:
            self.create_particle(90 + v, x, y)

        for i in range(0, len(self.particles)):
            # velocity, theta, x, y, ttl
            self.particles[i][4] -= 1  # decrement the ttl
            r = (self.particles[i][1] * math.pi) / 180
            self.particles[i][2] += (self.particles[i][0] * math.cos(r))
            self.particles[i][3] += (self.particles[i][0] * math.sin(r))

        self.clean_particles(all=False)

        # translate & rotate all particles
        for i in range(0, len(self.p_obj)):
            self.trans[i].set_translation(self.x_thresh + self.particles[i][2],
                                          self.y_thresh + self.particles[i][3])  # translate particle
            self.trans[i].set_rotation(self.particles[i][1])

    def create_particle(self, theta, x, y):
        p = [self.p_velocity, theta, x, y, self.p_ttl]
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

        self.clean_particles(all=False)
        return p

    def clean_particles(self, all):
        while self.particles and (all or self.particles[0][4] < 0):
            self.p_obj[0].set_color(self.bg_color[0], self.bg_color[1], self.bg_color[2])  # sets color of particle
            self.particles.pop(0)  # delete particle at beginning of list
            self.p_obj.pop(0)  # position of particle in list
            self.trans.pop(0)  # position of particle in list
