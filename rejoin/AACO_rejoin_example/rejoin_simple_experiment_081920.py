#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
# get rid of tensorflow annoying compatability warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# ### Setup Ray Temp Directory
# This will probably be different depending on your environment.

# In[ ]:


#temp_dir = '/home/aaco/tmp/ray'  # for docker container
temp_dir = '~/tmp/ray'


# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict, OrderedDict
import os
import math
import warnings

import gym
from gym import spaces
import numpy as np
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.tune.logger import pretty_print
from scipy.spatial.transform import Rotation as R
import utm
import yaml

from act3.agents.utilities.plugin_loader import get_integration_base
from act3.agents.utilities.util import expand_path, mkdir_p
from act3.environments.domain_randomization.dr_module import DomainRandomization
from act3.simulators.afsim.afsim_integration_controls import (
    AfsimStickAndThrottleP6DOFController,
)
from act3.simulators.afsim.afsim_integration_sensors import (
    AfsimFuelSensor,
    AfsimGloadSensor,
    AfsimOrientationRateSensor,
    AfsimSpeedKcasSensor,
)
from act3.simulators.base_integration import BaseIntegration, BasePlatform

ray.shutdown()
ray.init(ignore_reinit_error=True, log_to_driver=False, temp_dir=temp_dir, webui_host='127.0.0.1')


# ###  Environment Class
# This is a simple environment class. It interfaces with AFSIM and uses the Domain Randomization module to initialize the scenerio on each reset.

# In[ ]:


# Environment Class
class RejoinDREnv(gym.Env):
    """Simple rejoin task testing environment using AFSIM 2.6."""
    
    def __init__(self, env_config=None):
        """Initialize environment with provided configuration.
        
        The configuration specifies how long an episode should be
        in steps, what the step reward value is, the configuration
        for the simulation and specification for the initialization
        module that sets the state of the environment at the start
        of each episode.
        """
        # ******* Setup environment stuff ******
        # TODO: This will probably change
        
        # setup info dictionary
        self._info = {}
        
        # maximum length of episode
        if 'horizon' in env_config.keys():
            self._horizon = env_config['horizon']
        else:
            self._horizon = 1000
            
        self._info['horizon'] = self._horizon
            
        # setup for tracking episode length
        self._current_step = 0
        
        # setup step reward
        if 'step_reward' in env_config.keys():
            self._step_reward = env_config['step_reward']
        else:
            self._step_reward = 0.001
            
        self._info['step_reward'] = self._step_reward
            
        # TODO: this is probably in the env_config somewhere
        self.range_target = 6000 / 3.2808  # target range difference in meters
        self.alt_target = 0 / 3.2808  # target altitude difference in meters
        
        # ******* End setup environment stuff ******
        
        # ****** Start Simulation Setup ******
        # TODO: This should be a module that wraps the
        # AFSIM/PAW stuff, that way we could use this environment
        # directly with another simulator (e.g. Unity) and just
        # define an abstract class that enforces the interface
        
        # set the AFSIM simulator
        if 'sim_config' not in env_config.keys():
            raise (ValueError, 'Environment config must contain a sim_config!')
        self._sim_config = env_config['sim_config']
        
        base_output_path = env_config['sim_config']['scen']['output_path']
        try:
            worker_index = str(env_config.worker_index).zfill(4)
        except AttributeError:
            worker_index = str(0).zfill(4)
            
        self._output_path = os.path.join(base_output_path,
                                         worker_index)
        
        mkdir_p(self._output_path)
        
        self._sim_config['scen']['output_path'] = self._output_path
      
        # setup controller for afsim, this sets up basic hotas commands
        self._sim_config["platform_configs"] = {
            "blue0": {
                "controller_class_map": {
                    "mover": [(AfsimStickAndThrottleP6DOFController, {}),]
                },
                "sensor_class_map": {
                    "None": [
                        (AfsimOrientationRateSensor, {}),
                        (AfsimFuelSensor, {}),
                        (AfsimSpeedKcasSensor, {}),
                        (AfsimGloadSensor, {}),
                    ]
                },
            }
        }
        
        # AFSIM 2.6
        sim_class = get_integration_base(env_config['simulator'])
        self._sim = sim_class(config=self._sim_config)
       
        # TODO: Set the framerate of the simulation here
        """
        if 'frame_rate' in self._sim_config['scen'].keys():
            self._sim.update_rate = 1.0 / self._sim_config['scen']['frame_rate']
        else:
            self._sim.update_rate = (1.0 / 10.0)
        """
        
        self._sim.reset_simulation(self._sim_config)
        
        # ******* End Simulation Setup *******
         
        # ******* Setup Helper Classes *******
        self._current_episode = 0
        self._info['current_episode'] = self._current_episode
            
        # setup AFSIM to rllib observation converter
        self._obs_processor = AfsimStateToObs()
        
        # setup domain randomization module
        self._initializer = RejoinInitialConditions(env_config)
        self._info['phase'] = self._initializer.phase
        
        # setup action processor
        self._action_processor = AfsimHOTASController(env_config)
        
        # setup reward calculation module
        self._reward_calculator = RejoinRewardCalculator(env_config)
        
        # setup done checker module
        self._done_checker = RejoinDoneChecker(env_config)
        
        # ******* Setup Helper Classes *******
        
        # *** setup action and observation spaces ***
        self._setup_spaces()
        
    def set_phase(self, phase):
        """Set phase for UDR module.
        
        The phase governs how the initial states of the agent
        and lead aircraft are set at the start of each episode.
        """
        self._info['phase'] = phase
        self._initializer.set_phase(phase) 
       
    def _setup_spaces(self):
        """Setup action and observation spaces.
        
        These spaces are generated by the respective helper
        classes.
        """
        self.action_space = self._action_processor.get_action_space()
        
        action_obs_space = self._action_processor.get_observation_space()
        
        self.observation_space = self._obs_processor.get_observation_space(action_obs_space)
        
    def step(self, action):
        """Peform a single step of the environment.
        
        This will take an action generated by the policy and
        apply it to the agent in the simulation via the
        action processor helper class. The done conditions
        for the agent will then be checked and the reward
        for the agent's action will be generated based on
        the current state of the environment. Finally,
        the observation, reward, done and environment info
        will be returned to the policy for learning.
        
        Parameters
        ----------
        action : tuple[float]
            Action computed by the rllib policy
        
        Returns
        -------
        Tuple[dict, dict, dict, dict]
            Tuple containing [observation, reward, done, info]
        """
        # apply the action to the simulator
        platforms = self._sim.get_state()
        
        self._action_processor(platforms, action)
       
        sim_time = self._sim.advance_update()  # advance one step
        self._current_step += 1
        
        self._info['current_step'] = self._current_step
        
        # TODO: Does this need to happen? or are the platforms
        # just references?
        # try:
        # if ^ doesn't fail, then next line not needed
        sim_platforms = self._sim.get_state()
        #assert platforms == sim_platforms
        # if ^ doesn't fail, then next line not needed
       
        sim_state = {}
        for p in sim_platforms:
            sim_state[p.name] = p
            
        # might not need above for loop based on above test
            
        # popluate info with the current simulation state as
        # a StateDict
        # TODO: This should just be a dictionary to decouple
        # it from AFSIM-specific stuff
        # NOTE: the PAW encrypts the platform objects, so they
        # can't be pickled, if added to info they break everything
        # self._info['sim_state'] = sim_state
       
        self._info['action_state'] = self._action_processor.get_observation(sim_platforms)
        
        # TODO: add unscaled observation to info
        # get scaled observations
        obs = self._obs_processor(sim_state=sim_platforms,
                                  sim_time=sim_time,
                                  info=self._info)
        
        # process done state
        done = self._done_checker(self._info)
        
        # calculate reward from environment info
        reward = self._reward_calculator(self._info)
        
        return obs, reward, done, self._info
        #return obs, reward, done, {}
    
    def reset(self):
        """Reset environment and simulation state.
        
        This will archive the AFSIM aer file generated for the
        current episode. It will also reset all helper classes
        to their initial states.
        It will then reset the current state of the simulation
        environment and set the initial state of each platform
        in the environment based on the domain randomization
        module contained in self._initializer.
        """
       
        # TODO: This should be a part of the simulation module
        # as it is afsim specific
        # archive aer file
        temp = os.path.join(self._output_path, "replay.aer")
        if os.path.isfile(temp):
            os.rename(
                temp,
                os.path.join(
                    self._output_path,
                    (f"replay_{str(self._current_episode).zfill(8)}_"
                     f"{str(self._current_step).zfill(8)}.aer"),
                ),
            )
            
        # reset AFSIM 
        self._sim.reset_simulation(self._sim_config,
                                   self.update_platform_init)
        
        # reset observation processor
        self._obs_processor.reset()
        
        # reset reward calculator
        self._reward_calculator.reset()
        
        # reset done checker
        self._done_checker.reset()
       
        # get state information
        sim_time = 0.0
        sim_platforms = self._sim.get_state()
        
        # reset step counter
        self._current_step = 0
        self._info['current_step'] = self._current_step
       
        # increment episode counter
        self._current_episode += 1
        self._info['current_episode'] = self._current_episode
        
        # initialize action state
        self._info['action_state'] = self._action_processor.get_observation(sim_platforms)
        
        # reset num steps inside bubble counter
        self._info['num_steps_inside_bubble'] = 0
        
        return self._obs_processor(sim_state=sim_platforms,
                                   sim_time=sim_time,
                                   info=self._info)
    
    def update_platform_init(self, state):
        """Update sim configuration based on the currently set phase.
        
        This will position the agent and the lead aircraft, orient them
        and set their speeds based on the configuration and phase information
        processed by the initializer helper class.
        """
        self._initializer(state)
    
    @property
    def simulator(self):
        """The simulator used by this environment."""
        return self._sim


# ### Observation Processor
# 
# This takes the state of the simulation and provides an observation dictionary that can be ingested by rllib's policy network. This class is used by RejoinDREnv and handles all observation related processing.

# In[ ]:


# afsim observation to (scaled) rllib observation
class AfsimStateToObs:
    """Convert AFSIM 2.6 simulation state to rllib observation."""
    
    def __init__(self):
        """Setup observation processor state variables.
        
        Currently this just initializes the internal alt_diff
        variable that stores the altitude difference and time
        from the previous conversion.
        """
        self.reset()
       
        # scale factors
        # TODO: These values should be in a config or something
        self._scale_factors = {}
        self._scale_factors['altitude'] = [7500, 3000]
        self._scale_factors['velocity'] = [[0, 350], [0, 350], [0, 350]]
        # TODO: Figure out good values
        self._scale_factors['acceleration'] = [[0, 3500], [0, 3500], [0, 3500]]
        # TODO: Figure out good values
        self._scale_factors['angular_velocity'] = [[0, 3000], [0, 3000], [0, 3000]]
        # TODO: Adjust these - decrease variances?
        self._scale_factors['dist_from_goal'] = [[0, 10000], [0, 10000], [0, 7500]]
        # TODO: Figure out good values
        # TODO: these variances are too high
        self._scale_factors['dist_from_goal_rates'] = [[0, 10000], [0, 10000], [0, 7500]]
        # TODO: Same as dist from goal, should have a larger mean and smaller viance (isn't negative)
        self._scale_factors['range_to_goal'] = [0, 10000]
        # TODO: ensure this value is reasonable
        self._scale_factors['range_to_goal_rate'] = [0, 600]
        self._scale_factors['lead_rel_velocity'] = [[0, 100], [0, 100], [0, 100]]
        # TODO: this probably needs a non-zero mean and smaller variance, can't be negative
        self._scale_factors['range_to_lead'] = [0, 10000]
        
    def reset(self):
        """Reset state of the processor."""
        
        self._prev_time = 0.0
        self._prev_dist = 0.0
        self._prev_e_diff = 0.0
        self._prev_n_diff = 0.0
        self._prev_a_diff = 0.0
        self._prev_r2g = 0.0
        self._prev_ang_vel = [0.0, 0.0, 0.0]
   
    def __call__(self, sim_state, sim_time, info=None):
        """Get observation from simulation state.

        Parameters
        ----------
        TODO: Update this to reflect it is now a Tuple[AfsimPlatform, ...]
        sim_state : StateDict
            StateDict object containing entries for each entity in
            the simulation. For this environment, it is assumed to
            contain a blue0 (our agent) and red0 (the lead aircraft).
        sim_time : float
            Current time of the simulation.
        info : dict
            Information dictionary from the environment, optional
            
        Returns
        -------
        OrderedDict
            Dictionary of scaled observations
        """
        # get agent's state information
        platforms = {}
        for p in sim_state:
            platforms[p.name] = p
            
        current_time_diff = sim_time - self._prev_time
            
        # TODO: Most of the below should be refactored into functions
            
        blue = platforms['blue0']
        
        blue_alt = blue.position[2]
        # get orientation of aircraft in yaw, pitch and roll in radians
        blue_orientation = blue.orientation
        blue_quat = np.array(self.euler_to_quat(
            yaw=blue_orientation[0],
            pitch=blue_orientation[1],
            roll=blue_orientation[2],
            degrees=False))
        
        # get agent's control state
        if info is not None:
            blue_ctrls = info['action_state']
        else:
            raise (ValueError, "info parameter is required for this processor!")
        # -- old control stuff below --
        #raw_ctrls = blue.controllers[0].get_applied_control()
        # TODO: This should be handled by the action processor
        #blue_ctrls = [raw_ctrls[0],
        #              raw_ctrls[1],
        #              raw_ctrls[2],
        #              raw_ctrls[3] - 1]
        
        # get red's state information
        red = platforms['red0']
        
        # TODO: modify this to be randomly set at each reset
        # get goal location in UTM + altitude coordinates
        goal_e, goal_n, goal_alt = get_goal_position(
            platforms,
            rel_bearing=0.0,
            rel_range=6000.0/3.28084,  # 6 kft
            rel_alt=300.0/3.28084)  # 300 ft
        
        # get agent's position in UTM coordinates
        blue_ll = (blue.position[0], blue.position[1])
        blue_e, blue_n, z, l = utm.from_latlon(blue_ll[0], blue_ll[1])
        
        # find utm coordinate of this distance
        red_ll = (red.position[0], red.position[1])
        # convert to utm
        red_e, red_n, _, _ = utm.from_latlon(red_ll[0], red_ll[1], z, l) 
        
        # get distance from blue to red
        dist_r2b = np.sqrt((red_e - blue_e)**2 +
                           (red_n - blue_n)**2 +
                           (red.position[2] - blue.position[2])**2)
        
        # get distances from goal position
        e_diff = blue_e - goal_e
        n_diff = blue_n - goal_n
        a_diff = blue.position[2] - goal_alt
       
        # get difference of distances from last time step
        current_e_diff = abs(e_diff) - self._prev_e_diff
        current_n_diff = abs(n_diff) - self._prev_n_diff
        current_a_diff = abs(a_diff) - self._prev_a_diff
       
        # cache current difference for next frame
        self._prev_e_diff = abs(e_diff)
        self._prev_n_diff = abs(n_diff)
        self._prev_a_diff = abs(a_diff)
        
        # get range from goal position
        range_to_goal = np.sqrt((e_diff)**2 + (n_diff)**2 + (a_diff)**2)
        
        # get difference of range from last time step
        current_r2g = range_to_goal - self._prev_r2g
       
        # cache range for next frame
        self._prev_r2g = range_to_goal
       
        # get change rates of all distances
        if current_time_diff == 0:
            e_diff_rate = 0
            n_diff_rate = 0
            a_diff_rate = 0
            r2g_rate = 0
        else:
            e_diff_rate = current_e_diff / current_time_diff
            n_diff_rate = current_n_diff / current_time_diff
            a_diff_rate = current_a_diff / current_time_diff
            r2g_rate = current_r2g / current_time_diff
        
        
        # Get angular rates of agent
        blue_ang_vel = get_afsim_angular_rates(blue)
        
        # TODO: Get angular acceleration using finite difference (as function)
        # something like
        blue_ang_accel = get_afsim_angular_accel(blue_ang_vel,
                                                 self._prev_ang_vel,
                                                 current_time_diff)
        
        self._prev_ang_vel = blue_ang_vel
        
        self._prev_time = sim_time
        
        # get raw observations
        raw_obs = {}
        raw_obs['pose'] = blue_quat
        raw_obs['altitude'] = blue_alt
        raw_obs['controls'] = blue_ctrls
        raw_obs['velocity'] = blue.velocity_ned
        raw_obs['acceleration'] = blue.acceleration_ned
        raw_obs['angular_velocity'] = blue_ang_vel
        raw_obs['dist_from_goal'] = [e_diff, n_diff, a_diff]
        raw_obs['dist_from_goal_rates'] = [e_diff_rate, n_diff_rate, a_diff_rate]
        raw_obs['range_to_goal'] = range_to_goal
        raw_obs['range_to_goal_rate'] = r2g_rate
        raw_obs['lead_rel_velocity'] = red.velocity_ned - blue.velocity_ned
        raw_obs['range_to_lead'] = dist_r2b
        
        # add raw observations to info
        if info is not None:
            info['raw_observation'] = raw_obs
        else:
            warnings.warn("Info not provided, raw observations will not be stored",
                          RuntimeWarning)
        
        # scale observations and return
        obs = OrderedDict()
        for k, v in raw_obs.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                new_obs = []
                for i, o in enumerate(v):
                    if k in self._scale_factors.keys():
                        new_obs.append(self.scale_obs(o, self._scale_factors[k][i]))
                    else:
                        new_obs.append(o)
                obs[k] = np.array(new_obs)
            else:
                obs[k] = np.array([self.scale_obs(v, self._scale_factors[k])])
        
        return obs
    
    def scale_obs(self, obs, scale_factor=[0, 1]):
        """Scale observation by subtracting mean and dividing by variance.
        
        Parameters
        ----------
        obs : float
            Observation to scale
        
        scale_factor : [float, float]
            Tuple containing mean in [0] and variance in [1]
        
        Returns
        -------
        float
            Scaled observation
        """
        return (obs - scale_factor[0]) / scale_factor[1]
    
    # euler to quaternion
    def euler_to_quat(self, yaw, pitch, roll, degrees=True):
        """Convert yaw, pitch and roll rotation to quaternion.

        This function assumes that the angles are applied in
        yaw, pitch, roll order.

        Parameters
        ----------
        yaw : float
            Rotation about the Y axis in degrees/radians
        pitch : float
            Rotation about the X axis in degrees/radians
        roll : float
            Rotation about the Z axis in degrees/radians
        degrees : bool
            If True, angles are assumed to be in degrees,
            if false angles are assumed to be in radians
        """
        seq = 'yxz'
        angles = [yaw, pitch, roll]

        rot = R.from_euler(seq=seq, angles=angles, degrees=degrees)

        return rot.as_quat()
    
    # TODO: Add new observations (goal diff e, n, alt, vel/acc and angular vel/acc)
    # and remove unused ones (red_rel_pose, etc.)
    def get_observation_space(self, action_obs_space):
        """Get rllib observation space for the output of this processor.
        
        Parameters
        ----------
        action_obs_space : spaces
            Space of the observation component of the controls
            used in the simulator. This isn't necessarily the
            same as the action itself (e.g. for delta actions,
            the action space is a delta, but we'd want to use
            the actual control state of the aircraft in the
            simulator).
        """
        
        obs_space = OrderedDict({
           'pose': spaces.Box(-1, 1, shape=(4,)),
           'altitude': spaces.Box(-5, 5, shape=(1,)),
           # TODO: Probably need to pass in the action processor so it can
           # this can be pulled in directly
           #'controls': spaces.Box(low=np.array([-1, -1, -1, -1]),
           #                       high=np.array([1, 1, 1, 1]),
           #                       dtype=np.float32),
           'controls': action_obs_space,
           'velocity': spaces.Box(-10, 10, shape=(3,)),
           'acceleration': spaces.Box(-10, 10, shape=(3,)),
           'angular_velocity': spaces.Box(-10, 10, shape=(3,)),
           'dist_from_goal': spaces.Box(-10, 10, shape=(3,)),
           'dist_from_goal_rates': spaces.Box(-10, 10, shape=(3,)),
           'range_to_goal': spaces.Box(-10, 10, shape=(1,)),
           'range_to_goal_rate': spaces.Box(-10, 10, shape=(1,)),
           'lead_rel_velocity': spaces.Box(-10, 10, shape=(3,)),
           'range_to_lead': spaces.Box(-10, 10, shape=(1,)),
        })
        
        return spaces.Dict(obs_space)


# #### Goal Position Helper Function
# 
# This helper function computes the rejoin goal location in UTM and Altitude coordinates.

# In[ ]:


"""
Current rejoin success parameters:

CheckRejoinDone:
    # The radius in feet that the targe join need to occur, by default 100
    radius : 750
    # The time rate in ft/s for range change, by default 10
    capture_rate : 32
    # the target az from the lead, by default 0
    target_az : 0
    # the target slant range from the lead, by default 6000
    target_slant_range : 6000
    # The target elevation from the lead, by default 300
    target_elevation : 300
    # Time to check that rejoin has happend in seconds, by default 2
    time_in_bubble : 2
"""
# TODO: This is pretty inefficient and could be refactored into a class at some point
def get_goal_position(sim_state,
                      rel_bearing=0.0,
                      rel_range=6000,
                      rel_alt=300):
    """Calculate goal location from state of lead aircraft in ENU.
    
    This assumes East is the z axis of the aircraft (e.g forward) and
    North is the x axis of the aircraft (e.g. right). Heading is assumed
    to be 0 when pointing directly East.
    TODO: This convention might need to be flipped if the heading
    is different - need to confirm ENU coordinate frame
    
    Parameters
    ----------
    sim_state : StateDict
        StateDict object containing entries for each entity in
        the simulation. For this environment, it is assumed to
        contain a blue0 (our agent) and red0 (the lead aircraft).
        
    rel_bearing : float
        Bearing from the goal position to the lead aircraft in
        degrees
    
    rel_range : float
        Range in East, North plane from goal position to the
        lead aircraft in meters
        
    rel_alt : float
        Altitude difference between goal position and the
        lead aircraft
        
    Returns
    -------
    Tuple[float, float, float]
        A tuple containing the goal UTM easting location, the goal
        UTM northing location and the goal altitude
    """
    # get lead aircraft's heading
    heading = sim_state['red0'].orientation[0]
    # convert to degrees
    heading *= 180 / math.pi
    
    # get angle from rejoin point to lead
    angle = ((rel_bearing + heading) % 360)
    
    # find e, n location in meters
    e = rel_range * math.sin(math.radians(angle))  # north is heading = 0.0
    n = rel_range * math.cos(math.radians(angle))  # north is heading = 0.0
    #e = rel_range * math.cos(math.radians(angle))  # east is heading = 0.0
    #n = rel_range * math.sin(math.radians(angle))  # east is heading = 0.0
        
    # get agent's utm zone
    blue_ll = (sim_state['blue0'].position[0],
               sim_state['blue0'].position[1])
    blue_e, blue_n, z, l = utm.from_latlon(blue_ll[0], blue_ll[1])
    # find utm coordinate of this distance
    red_ll = (sim_state['red0'].position[0], sim_state['red0'].position[1])
    # convert to utm
    red_e, red_n, _, _ = utm.from_latlon(red_ll[0], red_ll[1], z, l) 
    
    goal_e = red_e - e
    goal_n = red_n - n
    
    # get altitude of goal position
    goal_alt = sim_state['red0'].position[2] + rel_alt
    
    return goal_e, goal_n, goal_alt


# #### Get Angular Rates from AfsimPlatform Helper Function

# In[ ]:


def get_afsim_angular_rates(platform):
    """Get angular rates from afsim platform.

    Parameters
    ----------
    platform : AfsimPlatform

    Returns
    -------
    Tuple[float, float, float]
        Yaw, pitch and roll rates
    """
    mover, _, _, _ = platform._platform.get_components()
    
    return mover.get_yaw_rate(), mover.get_pitch_rate(), mover.get_roll_rate()

# TODO: Implement angular accelerations function
def get_afsim_angular_accel(ang_vel,
                            prev_ang_vel,
                            time_diff):
    """Estimate angular acceleration through finite difference method.
    
    Parameters
    ----------
    ang_vel : [float, float, float]
        Angular velocity of current frame
    prev_ang_vel : [float, float, float]
        Angular velocity at time - time_diff
    time_diff : float
        Difference in seconds between measurements
    """
    if time_diff > 0:
        return [(t - p) / time_diff for t, p in zip(ang_vel, prev_ang_vel)]
    else:
        return [0.0, 0.0, 0.0]


# ### Create Support Classes
# 
# Here we create helper classes that will calculate the reward for each step, determine if the agent is done and an action processor that will take the action output by the rllib policy network and apply it to the AFSIM simulator.
# 
# Current there is a reward calculator class, a done checking class and an action processor class.

# In[ ]:


class RejoinRewardCalculator:
    """Class to compute rewards for agent performing rejoin task."""
    def __init__(self, config):
        """Initialize the calculator.
        
        Parameters
        ----------
        config : dict
            Dictionary containing parameters for the 
            various reward calculations performed by
            this object
        """
        # TODO: Grab all this from config
        self._step_reward = 0.0005
       
        # acceptable distance from goal to be in success 'bubble'
        self._bubble = [100, 100, 100]
        
        # warning zones
        self._min_alt = 500
        self._max_alt = 45000
        self._min_range = 300
        self._max_range = 18000
        
    def reset(self):
        """Reset the state of this object."""
        # TODO: Return all parameters to their initial values,
        # this probably won't do anything
        
    def __call__(self, info):
        """Compute the reward given the environment information.
        
        Parameters
        ----------
        info : dict
            Dictionary containing information on the state of
            the environment, scenario, agent, etc. 
            TODO: What should be in here for this specific
            object?
        """
        reward = 0
       
        # TODO: scale lateral and altitude changes seperately
        
        # if moving toward goal, award self._step_reward
        #if info['raw_observation']['range_to_goal_rate'] <= 0:
            #reward = self._step_reward
        reward = -0.00001 * info['raw_observation']['range_to_goal_rate']
                
        # if inside success bubble, award 2 * self._step_reward
        # NOTE: bubble is pretty small right now
        # TODO: Make sure this is much larger than the distance toward
        # goal reward, basically should be 2x the largest distance toward
        # reward
        dists = info['raw_observation']['dist_from_goal']
        if (abs(dists[0]) <= self._bubble[0] and
                abs(dists[1]) <= self._bubble[1] and
                abs(dists[2]) <= self._bubble[2]):
            #reward = 2 * self._step_reward
            reward = 10 * self._step_reward
            info['num_steps_inside_bubble'] += 1
        
        # if nearing mission altitude bounds, penalize
        alt = info['raw_observation']['altitude']
        if (alt < self._min_alt) or (alt > self._max_alt):
            reward = -3 * self._step_reward
       
        # if nearing mission range bounds, penalize
        dist = info['raw_observation']['range_to_lead']
        if (dist < self._min_range) or (dist > self._max_range):
            reward = -3 * self._step_reward
            
        return reward
        
class RejoinDoneChecker:
    """
    Class to check of the agent has met done conditions
    for the Rejoin task.
    """
    def __init__(self, config):
        """Initialize the calculator.
        
        Parameters
        ----------
        config : dict
            Dictionary containing parameters for the 
            various reward calculations performed by
            this object
        """
        # TODO: actually use a configuration
        self._min_alt = 100
        self._max_alt = 50000
        
        self._min_range = 100
        self._max_range = 20000
        
    def reset(self):
        """Reset the state of this object."""
        # TODO: Return all parameters to their initial values,
        # might not need anything
        
    def __call__(self, info):
        """
        Determine if the agent is done based on the information
        provided on the state of the evironment.
        
        Parameters
        ----------
        info : dict
            Dictionary containing information on the state of
            the environment, scenario, agent, etc. 
            TODO: What should be in here for this specific
            object?
        """
        # Notes: right now this can just check to see if the aircraft
        # is outside some mission bounds as defined in the config and
        # if the number of steps is greater than horizon. Could return
        # the done state in the info if desired.
        step_done = info['current_step'] > info['horizon']
        
        # check if agent is outside altitude bounds
        alt = info['raw_observation']['altitude']
        alt_done = alt < self._min_alt or alt > self._max_alt
       
        # check if agent is outside of range bounds
        # TODO: Get range to lead aircraft for min range check
        dist = info['raw_observation']['range_to_lead']
        dist_done = dist < self._min_range or dist > self._max_range
        
        done_states = {'step_done': step_done,
                       'alt_done': alt_done,
                       'dist_done': dist_done}
        
        info['done_states'] = done_states
        
        if (step_done or alt_done or dist_done):
            done = True
        else:
            done = False
        
        return done
    
class AfsimHOTASController:
    """Class to apply HOTAS commands to AFSIM.
    
    This class handles all aspects of the action. It applies the
    action supplied by the policy to the simulation, provides the
    action space to the policy and provides the observation and
    observation space to the observation processor if state of the
    control is desired to be a part of the observation.
    """
    def __init__(self, config):
        """Setup the control processor.

        Parameters
        ----------
        config : dict
            Dictionary containing the names of the agents
            that this controller will send commands for
            TODO: should this be env_config???
        """
        # TODO: actually use a configuration
        self._agent_name = 'blue0'

    def get_action_space(self):
        """Get the action space of this object.

        Returns:
            spaces.Box
            The action space that this object uses
        """
        return spaces.Box(low=np.array([-1, -1, -1, -1]),
                          high=np.array([1, 1, 1, 1]),
                          dtype=np.float32)
    
    def get_observation_space(self):
        """Get the observation space of the controller.

        Returns:
            spaces.Box
            The observation space that this object uses
        """
        return spaces.Box(low=np.array([-1, -1, -1, -1]),
                          high=np.array([1, 1, 1, 1]),
                          dtype=np.float32)
    
    def get_observation(self, platforms):
        """Get the observation of the controller.
        
        Parameters
        ----------
        platforms : Tuple[AfsimPlatform, ...]
            Tuple of AfsimPlatform objects, with an
            entry for each platform in the scene
            
        Returns
        -------
        Tuple[float, float, float, float]
            The state of the control of the platform.
        """
        ctrl = None
        for p in platforms:
            if p.name == self._agent_name:
                ctrl = p.controllers[0].get_applied_control()
       
        if ctrl is None:
            raise (ValueError, f"Control not available for {self._agent_name}!")
            
        agent_ctrls = [ctrl[0],
                       ctrl[1],
                       ctrl[2],
                       ctrl[3] - 1]
        
        return agent_ctrls

    def __call__(self, platforms, action):
        """Process the rllib action and apply in AFSIM.

        Parameters
        ----------
        platforms : Tuple[AfsimPlatform, ...]
            Tuple of AfsimPlatform objects, with an
            entry for each platform in the scene 
            
        action : np.array
            Numpy array of action calculation from rllib's
            agent.compute_action method
        """
        action = flatten_to_single_ndarray(action)
        # get action for afsim
        aileron = action[0]
        elevator = action[1]
        rudder = action[2]
        throttle = (action[3] + 1) / 2  # no afterburner
        
        sim_action = {self._agent_name: [aileron, elevator, rudder, throttle]}
       
        # Note: this is pretty simple because we only have one controller,
        # if there were multiple, we would need a second for loop to loop
        # over the controllers and get the name of the one we care about
        for p in platforms:
            if p.name == self._agent_name:
                ctrl = p.controllers[0]
                ctrl.apply_control(np.array(sim_action[self._agent_name]))


# ### Initializer
# 
# Here we define a helper class that will generate random starting states for the agent and lead platforms in the environment. We can use this at each reset to randomly initialize each aircraft according to rules defined in the env_config. This is a standalone version of what is embedded into the RejoinChallenge and base AACO Environment classes.

# In[ ]:


import afsim

from act3.environments.util.env_common import EnvCommonValues
from act3.environments.util.platform_centric_utils import PlatformCentricUtils


class RejoinInitialConditions:
    """Class to set starting state of AFSIM platforms."""
    
    _BLUE0 = "blue0"
    _RED0 = "red0"
    _LEAD_LAT = "lead_lat"
    _LEAD_LON = "lead_lon"
    _LEAD_ALT = "lead_alt"
    _LEAD_HEADING = "lead_heading"
    _LEAD_HEADING_DEF = 180.0
    _LEAD_MAN_ROLL = "lead_manuever_roll"
    _LEAD_ROLL_MAN_DEF = 0
    _LEAD_KCAS_MAN = "lead_manuever_kcas"
    _LEAD_KCAS_MAN_DEF = 350
    _FOLLOW_HEADING = "follow_heading"
    _SET_ROLL_ANGLE_CMD = "SetRollAngle"
    _SET_ALTITUDE_CMD = "SetAltitude"
    _LEAD_ALT_MAN = "lead_manuever_altitude"
    _SET_KCAS_CMD = "SetKCAS"
    _DEFAULT_VELOCITY_NED = 180.0  # m/s : 350 KCAS
    _TRAIL_ALT_DIFF = "trail_alt_diff"
    _TRAIL_ALT_DIFF_DEF = 0
    _START_RANGE = "start_range"
    _START_RANGE_DEF = 11482
    _TRAIL_AZIMUTH = "trail_azimuth"
    _TRAIL_AZIMUTH_DEF = 0
    
    def __init__(self, env_config):
        """Initialize udr module"""
        self._env_config = env_config
        self.dr_module = DomainRandomization(env_config)
        self.phase = env_config['domain_randomization_config']['randomizing_phase']
        
    def set_phase(self, phase):
        """Set the phase of the domain randomization module.
        
        Parameters
        ----------
        phase : int
            Phase of the domain randomization module to use.
        """
        env_config = self._env_config
        env_config['domain_randomization_config']['randomizing_phase'] = phase
        self.phase = phase
        self.dr_module = DomainRandomization(env_config)
    
    def __call__(self, state):
        """Set the initial state of the aircraft.
        
        Parameters
        ----------
        state : Tuple[AfsimPlatform, ...]
            Tuple of AfsimPlatform objects, with an
            entry for each platform in the scene
        """
        params = self.dr_module.sample()
        lead_lat = params[self._LEAD_LAT]
        lead_lon = params[self._LEAD_LON]
        lead_alt_f = params[self._LEAD_ALT]
        lead_alt_m = lead_alt_f * EnvCommonValues.FEET_TO_METERS
        red_heading_deg = params.get(self._LEAD_HEADING, self._LEAD_HEADING_DEF)
        red_heading_rad = afsim.Util.degrees_to_radians(red_heading_deg)
        lead_roll_angle_deg = params.get(self._LEAD_MAN_ROLL, self._LEAD_ROLL_MAN_DEF)
        lead_manuever_kcas = params.get(self._LEAD_KCAS_MAN, self._LEAD_KCAS_MAN_DEF)
        # currently not in env_config.yml phase 1-8, will always use lead_alt_f
        lead_manuever_alt = params.get(self._LEAD_ALT_MAN, lead_alt_f)
        b0_alt_diff = self.get_b0_alt_diff(params)
        b0_slant_range = self.get_b0_slant_range(params)
        b0_azimuth = self.get_b0_azimuth(params, red_heading_deg)

        blue_lat, blue_lon, _ = PlatformCentricUtils.aer2geodetic(
            az=b0_azimuth,
            el=0,
            srange=b0_slant_range,
            lat0=lead_lat,
            lon0=lead_lon,
            h0=lead_alt_m,
        )

        # blue0 defaults to pointing at red0, but may have it's own
        b0_heading_deg = params.get(self._FOLLOW_HEADING, (b0_azimuth - 180) % 360)
        b0_heading_rad = afsim.Util.degrees_to_radians(b0_heading_deg)
        for platform in state:
            p = platform._platform  # pylint: disable=W0212
            if platform.name == self._RED0:
                p.set_location_lla(lead_lat, lead_lon, lead_alt_m)
                vel_n, vel_e, vel_d = self.calculate_init_velocity_component(
                    red_heading_rad, self._DEFAULT_VELOCITY_NED
                )
                p.set_orientation_ned(red_heading_rad, 0, 0)
                p.set_velocity_ned(np.array([vel_n, vel_e, vel_d]))
                afsim.util.Util.update_execute_function(
                    p, self._SET_ALTITUDE_CMD, lead_manuever_alt
                )
                afsim.util.Util.update_execute_function(
                    p, self._SET_ROLL_ANGLE_CMD, lead_roll_angle_deg
                )
                afsim.util.Util.update_execute_function(
                    p, self._SET_KCAS_CMD, lead_manuever_kcas
                )

            else:
                p.set_location_lla(
                    blue_lat, blue_lon, lead_alt_m + b0_alt_diff
                )
                p.set_orientation_ned(
                    b0_heading_rad, 0, 0
                )  # TODO: get actual heading for blue
                vel_n, vel_e, vel_d = self.calculate_init_velocity_component(
                    b0_heading_rad, self._DEFAULT_VELOCITY_NED
                )
                p.set_velocity_ned(np.array([vel_n, vel_e, vel_d]))
    
    @staticmethod
    def calculate_init_velocity_component(heading: float, speed: float):
        """
        calculate_init_velocity_component takes in a heading a speed and then
        calculates the NED velocity components from a speed value

        Arguments:
            heading: float
                the heading direction for this aircraft in radians
            speed: float
                the speed for this aircraft
        Returns
        -------
        vel_n: float
            The north component in the velocity vector
        vel_e: float
            The east component of the velocity vector
        vel_d: float
            The down component of the velocity vector (for now always returns 0)
        """
        # THESE ARE REVERSED ON PURPOSE, vel_e is vel_n and viceversa because
        # of afsim orientation, N,E,S,W=0,90,180,270 so return opposite values
        vel_e = np.sin(heading) * speed
        vel_n = np.cos(heading) * speed
        return vel_n, vel_e, 0
    
    @staticmethod
    def get_b0_alt_diff(params: dict) -> float:
        """
        get_b0_alt_diff gets the altitude difference between the red0 and blue0
        agent

        Parameters
        ----------
        params : typing.Dict
            DR module env parameters,
            may contain "trail_alt_diff"

        Returns
        -------
        b0_alt_diff: float
            The Altitude difference between the red0 agent and the blue0 agent
        """
        b0_alt_diff = params.get(
            RejoinInitialConditions._TRAIL_ALT_DIFF, RejoinInitialConditions._TRAIL_ALT_DIFF_DEF
        )
        return b0_alt_diff * EnvCommonValues.FEET_TO_METERS

    @staticmethod
    def get_b0_slant_range(params: dict) -> float:
        """
        get_b0_slant_range gets the slant range between the red0 and blue0
        agent

        Parameters
        ----------
        params : typing.Dict
            DR module env parameters,
            may contain "start_range"

        Returns
        -------
        b0_slant_range: float
            The slant range difference between the red0 agent and the blue0 agent
        """
        b0_slant_range = params.get(
            RejoinInitialConditions._START_RANGE, RejoinInitialConditions._START_RANGE_DEF
        )
        return b0_slant_range * EnvCommonValues.FEET_TO_METERS

    @staticmethod
    def get_b0_azimuth(params: dict, red_heading_deg: float) -> float:
        """
        get_b0_azimuth gets the azimuth between the red0 and blue0
        agent

        Parameters
        ----------
        params : typing.Dict
            DR module env parameters,
            may contain "trail_azimuth"

        Returns
        -------
        b0_azimuth: float
            The azimuth difference between the red0 agent and the blue0 agent
        """
        b0_aspect_angle = params.get(
            RejoinInitialConditions._TRAIL_AZIMUTH, RejoinInitialConditions._TRAIL_AZIMUTH_DEF
        )
        azimuth = ((red_heading_deg - 180) % 360 + (b0_aspect_angle * -1)) % 360
        # pymap3d seems to take the abs of the azimuth
        # so need to get proper angle
        b0_azimuth = (360 + azimuth) % 360
        return b0_azimuth


# ## Create Policy 
# 
# Here we create the policy. First, we load in the AACO environment configurations yaml files.

# In[ ]:


# Simulation Configuration
# sim_config_yml = ('/proj/aaco/team/ben/projects/act3-rllib-agents/'
#                   'config/tasks/rejoin_UDR/sim_config.yml')
sim_config_yml = ('/proj/aaco/team/ben/projects/act3-rllib-agents/config/tasks/rejoin/sim_config.yml')
env_config_yml = ('/proj/aaco/team/ben/projects/act3-rllib-agents/config/tasks/rejoin/env_config.yml')


with open(sim_config_yml) as f:
    sim_config = yaml.safe_load(f)

from act3.agents.utilities.yaml_loader import Loader, construct_include

yaml.add_constructor("!include", construct_include, Loader)

with open(env_config_yml) as f:
    tmp_env_config = yaml.load(f, Loader)
    
#config = {'output_path': '~/ray_results'}
env_config = tmp_env_config['environment']
#env_config['sim_config'] = sim_config['scen']
env_config['sim_config'] = sim_config
env_config['sim_config']['scen']['frame_rate'] = 10  # set the frame rate of the simulation
# All output of training will be placed into the specified directory
# TODO: This used to place all of the checkpoints into this directory as well, but 
# that seems to have changed. Need to figure out how to fix that.
env_config['sim_config']['scen']['output_path'] = 'output_simple_rejoin_082620_128x128'


# In[ ]:


env_config['simulator'] = 'AfsimIntegration'  # this might not be needed any longer


# In[ ]:


# Environment Configuration
# set max number of steps for an episode
env_config['horizon'] = 1000
# set the step reward magnitude for the environment
#env_config['step_reward'] = 1.0 / float(env_config['horizon'])
env_config['step_reward'] = 0.4 / float(env_config['horizon'])


# In[ ]:


# Training Configuration
# policy training parameters for PPO
trainer_config = DEFAULT_CONFIG.copy()
trainer_config['num_workers'] = 12
trainer_config['train_batch_size'] = 12000
trainer_config['gamma'] = 0.995
trainer_config['lambda'] = 0.9  # from googlebrain paper
trainer_config['rollout_fragment_length'] = 1000
trainer_config['batch_mode'] = 'complete_episodes'
trainer_config['sgd_minibatch_size'] = 4000 
trainer_config['clip_param'] = 0.3
trainer_config['vf_clip_param'] = 10.0
trainer_config['model']['vf_share_layers'] = False  # from googlebrain paper
# trainer_config['num_sgd_iter'] = 10
trainer_config['num_sgd_iter'] = 30
trainer_config['lr'] = 5e-4


# In[ ]:


# this network is pretty compact, might need to be beefed up
# trainer_config['model']['fcnet_hiddens'] = [32, 32]  # from 8-26-20 (didn't seem to work well after 2 hours training)
trainer_config['model']['fcnet_hiddens'] = [128, 128]  # from 08-25-20 (output_simple_rejoin_test_v2)
# trainer_config['model']['fcnet_hiddens'] = [256, 256, 256]  # from 8-25-20 (output_simple_rejoin_256x256x256) did not look good
#trainer_config['model']['free_log_std'] = True
#trainer_config['model']['use_lstm'] = True  # use with smaller model [64, 64] or [32, 32]
#trainer_config['model']['max_seq_len'] = 50
#trainer_config['model']['lstm_use_prev_action_reward'] = True
# environment configuration
trainer_config['env_config'] = env_config
# probably don't need this stuff anymore
trainer_config['env_config']['sim_config']['scen']['blue_team'][0]['route'] = [[39.8537, -84.0537], [39.7137, -84.0537]]
trainer_config['env_config']['sim_config']['scen']['blue_team'][0]['position'] = [39.8537, -84.0537]


# ## Train the Policy With Backtracking and Reward Collapse Protection

# In[ ]:


# Setup Output Path for Checkpoints

# for local testing this will place checkpoints in the same parent folder as the aer files
# this probably needs to be changed for docker container to something like the below
#ray_results = '/home/aaco/ray_results/ACT3-RLLIB-AGENTS/'  # for docker container
ray_results = './'  
output_path = (ray_results + trainer_config['env_config']['sim_config']['scen']['output_path']
               + '/checkpoints')

# Check if path exists. If not create it
if not os.path.exists(output_path):
    os.makedirs(output_path)


# In[ ]:


# Create PPO Agent
trainer = PPOTrainer(config=trainer_config, env=RejoinDREnv)

# Restore old checkpoint if desired
# trainer.restore(output_path + '/checkpoint_900/checkpoint-900')

# Setup Training Parameters
initial_lr = trainer_config['lr']
initial_thresh = 1000
drop_thresh = 100
best_reward_mean = -10.0
counter = 0
thresh = 1000
min_lr = 5e-10
lr_decrement = 10
max_iter = 2e7
# max_iter = 5  # for testing
prev_worse = False
vcount = 0
i = 0

while trainer_config['lr'] >= min_lr and i < max_iter:
    print(f'Training iteration {i}...')
    result = trainer.train()
    ep_mean = result['episode_reward_mean']
    print(f'Episode reward mean is {ep_mean} for training iteration {i}...')
    # checkpoint model if mean reward is best so far
    if result['episode_reward_mean'] >= best_reward_mean:
        checkpoint_path = trainer.save(output_path)
        best_reward_checkpoint = checkpoint_path
        best_reward_mean = result['episode_reward_mean']
        print(f'Checkpoint {i} saved to {checkpoint_path}')
        print(pretty_print(result))
        counter = 0
    else:
        counter += 1
    # check to see if reward is falling
    if ((ep_mean - best_reward_mean) / abs(best_reward_mean)) < -0.2:
        print(f'Iteration {i} had a mean reward more than 20% worse '
              f'then the current best reward!')
        print(f'Mean reward was {ep_mean} and best is {best_reward_mean}...')
        if prev_worse:
            vcount += 1
        prev_worse = True
    else:
        prev_worse = False
        vcount = 0
        
    if trainer_config['lr'] == initial_lr:
        fall_thresh = initial_thresh
    else:
        fall_thresh = thresh
    # grace period for falling reward
    if (counter > fall_thresh) or (vcount > drop_thresh):
        print('Reducing learning rate!')
        print(f'Steps without reward increase was {counter}'
              f'and consecutive steps with reward drop was {vcount}')
        print('Resuming from previous best checkpoint and '
              'decrementing learning rate...')
        trainer_config['lr'] /= lr_decrement
        
        if vcount > drop_thresh: 
            # only backtrack if reward collapse was detected
            trainer = PPOTrainer(config=trainer_config, env=RejoinDREnv)
            trainer.restore(best_reward_checkpoint)
            vcount = 0
        else:
            # otherwise restore current state with lower learning rate
            checkpoint_path = trainer.save(output_path)
            trainer = PPOTrainer(config=trainer_config, env=RejoinDREnv)
            trainer.restore(checkpoint_path)
            vcount = 0
            
        counter = 0
    
    i += 1


# #### Save environment configuration as pickle file
# This mimics what ray does for its checkpoints and enables the environment to be used by the
# AACO evaluator.

# In[ ]:


import pickle

params_file = (trainer.config['env_config']['sim_config']['scen']['output_path'] + '/checkpoints/params.pkl')

trainer.config['env'] = 'RejoinDREnv-v0'

pickle.dump(trainer.config, open(params_file, "wb" ) )


# ## Try Evaluation
# 
# This might need to be updated. Haven't tried using it with the evaluator since June.

# In[ ]:


from act3.agents.evaluation import RejoinEvaluator


# In[ ]:


import pickle

#checkpoint_path = '/home/aaco/ray_results/ACT3-RLLIB-AGENTS/PPO_RejoinChallenge_0_2020-07-15_03-05-397mn8482f'
experiment_path = None
params_file = os.path.join(experiment_path,'params.pkl') 

params = pickle.load( open( params_file, "rb" ) )


# In[ ]:


# register environment
from ray.tune.registry import register_env

register_env("RejoinEnv-v0", lambda config: MatchAltEnv(config))


# In[ ]:


# Restart ray
ray.shutdown()
ray.init(ignore_reinit_error=True, log_to_driver=False, temp_dir=temp_dir, webui_host='127.0.0.1')


# In[ ]:


evaluator = RejoinEvaluator(experiment_path=experiment_path, ray_temp=temp_dir)


# In[ ]:


evaluator.set_phase(phase=2)

metrics = evaluator.evaluate(num_episodes=3, visualize_episodes=True)


# In[ ]:


len(metrics)


# In[ ]:


metrics[0].keys()


# In[ ]:


len(metrics[0]['blue0'])


# In[ ]:


metrics[2]['blue0']['num_steps']

