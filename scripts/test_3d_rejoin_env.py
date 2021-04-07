import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import math
import yaml

from datetime import datetime

import ray
from ray import tune

import ray.rllib.agents.ppo as ppo
from ray.tune.logger import JsonLogger

from saferl.aerospace.tasks.rejoin import *
from saferl.environment.tasks import *
from saferl.aerospace.tasks import *

from saferl.aerospace.models.dubins import Dubins3dPlatform
from saferl.environment import build_callbacks_caller, EpisodeOutcomeCallback, FailureCodeCallback, \
                                        RewardComponentsCallback, LoggingCallback, LogContents, RelativeCylinder

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='./output')

args = parser.parse_args()

expr_name = datetime.now().strftime("expr_%Y%m%d_%H%M%S")
output_dir = os.path.join(args.output_dir, expr_name)

# set logging verbosity options
num_logging_workers = 1
logging_interval = 10                        # log every 10th episode
contents = (LogContents.VERBOSE,)           # tuple of desired contents

def run_rollout(agent, env_config):
    # instantiate env class
    env = DubinsRejoin(env_config)

    info_histroy = []
    obs_history = []
    reward_history = []
    action_history = []

    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        info_histroy.append(info)
        obs_history.append(obs)
        reward_history.append(reward)
        action_history.append(action)

    rollout_data = {
        'episode_reward': episode_reward,
        'info_history': info_histroy,
        'obs_history': obs_history,
        'reward_history': reward_history,
        'action_history': action_history
    }

    return rollout_data


ray.init(num_gpus=0)
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 6
config['_fake_gpus'] = True
config['seed'] = 0
config['callbacks'] = build_callbacks_caller([EpisodeOutcomeCallback(),
                                              FailureCodeCallback(),
                                              RewardComponentsCallback(),
                                              LoggingCallback(num_logging_workers=num_logging_workers,
                                                              episode_log_interval=logging_interval,
                                                              contents=contents)])
config['output']=os.path.join(args.output_dir, expr_name)
config['output_max_file_size'] = 999999
# config['log_level'] = 'ERROR'
config['monitor'] = True

rollout_history = []


# # ------ Define reward configuration ------
#
# reward_processors = [
#     RejoinRewardProcessor,
#     RejoinFirstTimeRewardProcessor,
#     TimeRewardProcessor,
#     RejoinDistanceChangeRewardProcessor,
#     FailureRewardProcessor,
#     SuccessRewardProcessor
# ]
#
# reward_config = {
#     'processors': reward_processors,
#     'time_decay': -0.01,
#     # 'time_decay': -0.03,
#     'failure': {
#         'timeout': -1,
#         'crash': -1,
#         'distance': -1,
#     },
#     'success': 1,
#     'rejoin_timestep': 0.1,
#     'rejoin_first_time': 0.25,
#     # 'dist_change': -0.00001,
#     'dist_change': -0.0001,
# }
#
#
# # ------ Define status configuration ------
#
# status_processors = [
#     DubinsInRejoin,
#     DubinsInRejoinPrev,
#     DubinsRejoinTime,
#     DubinsTimeElapsed,
#     DubinsLeadDistance,
#     DubinsFailureStatus,
#     DubinsSuccessStatus
# ]
#
# status_config = {
#     'processors': status_processors,
#     'safety_margin': {
#         'aircraft': 100
#     },
#     'timeout': 1000,
#     'max_goal_distance': 100000,
#     'success': {
#         'rejoin_time': 20,
#     },
# }
#
# # ------ Define observation configuration ------
#
# observation_processors = [
#     Dubins3dObservationProcessor
# ]
#
# observation_config = {
#     'processors': observation_processors,
#     # 'mode': 'rect',
#     # 'reference': 'global',
#     'mode': 'magnorm',
#     'reference': 'wingman',
# }
#
#
# rejoin_config = {
#     'reward': reward_config,
#     'init': {
#         'wingman': {
#             # 'x': [-4000, 4000],
#             # 'y': [-4000, 4000],
#             'x': [0, 0],
#             'y': [0, 0],
#             'z': [0, 0],
#             # 'heading': [-math.pi/4, math.pi/4],
#             'heading': [0, 0],
#             # 'gamma': [-math.pi, math.pi],
#             'v': [500, 500]
#         },
#         'lead': {
#             'x': [2000, 2000],
#             'y': [0, 0],
#             'z': [0, 0],
#             # 'heading': [-math.pi/4, math.pi/4],
#             'heading': [0, 0],
#             'v': [550, 550]
#         },
#     },
#     'agent':{
#         'model': '3d',
#         'controller':{
#             'actuators': [
#                 {
#                     'name': 'ailerons',
#                     'space': 'discrete',
#                     'points': 5,
#                 },
#                 {
#                     'name': 'elevator',
#                     'space': 'discrete',
#                     'points': 5,
#                 },
#                 {
#                     'name': 'throttle',
#                     'space': 'discrete',
#                     'points': 5,
#                 },
#             ],
#         },
#     },
#     'observation': observation_config,
#     'rejoin_region': {
#         'type': 'cylinder',
#         'range': 500,
#         'aspect_angle': 60,
#         'radius': 150,
#         'height': 300
#     },
#     'status': status_config,
#     'verbose': False,
# }

# ------ new config style ------

rejoin_config = {
    "env_objs": [
        {
            "name": "wingman",
            "class": Dubins3dPlatform,
            "config": {
                "controller": {
                    "actuators": [
                        {
                            'name': 'ailerons',
                            'space': 'discrete',
                            'points': 5,
                        },
                        {
                            'name': 'elevator',
                            'space': 'discrete',
                            'points': 5,
                        },
                        {
                            'name': 'throttle',
                            'space': 'discrete',
                            'points': 5,
                        }
                    ]
                },
                "init": {
                    'x': [-4000, 4000],
                    'y': [-4000, 4000],
                    # 'x': [0, 0],
                    # 'y': [0, 0],
                    'z': [-1000, 1000],
                    'heading': [-math.pi/4, math.pi/4],
                    # 'heading': [0, 0],
                    'v': [300, 800]
                }
            },
        },
        {
            "name": "lead",
            "class": Dubins3dPlatform,
            "config": {
                "init": {
                    'x': [-4000, 4000],
                    'y': [-4000, 4000],
                    # 'x': [2000, 2000],
                    # 'y': [0, 0],
                    'z': [-1000, 1000],
                    'heading': [-math.pi/4, math.pi/4],
                    # 'heading': [0, 0],
                    'v': [550, 550]
                }
            },

        },
        {
            "name": "rejoin_region",
            "class": RelativeCylinder,
            "config": {
                'ref': 'lead',
                'track_orientation': True,
                'r_offset': 500,
                'aspect_angle': 60,
                'radius': 150,
                'height': 300,
                "init": {}
            },
        },
    ],
    "agent": "wingman",
    "status": [
        {
            "name": "in_rejoin",
            "class": DubinsInRejoin,
            "config": {
                "rejoin_region": "rejoin_region",
                "wingman": "wingman"
            }
        },
        {
            "name": "in_rejoin_prev",
            "class": DubinsInRejoinPrev,
            "config": {
                'rejoin_status': "in_rejoin",
            }
        },
        {
            "name": "rejoin_time",
            "class": DubinsRejoinTime,
            "config": {
                'rejoin_status': "in_rejoin",
            }
        },
        {
            "name": "time_elapsed",
            "class": DubinsTimeElapsed,
            "config": {}
        },
        {
            "name": "lead_distance",
            "class": DubinsLeadDistance,
            "config": {
                "wingman": "wingman",
                "lead": "lead"
            }
        },
        {
            "name": "failure",
            "class": DubinsFailureStatus,
            "config": {
                "lead_distance": "lead_distance",
                "time_elapsed": "time_elapsed",
                'safety_margin': {
                    'aircraft': 100
                },
                'timeout': 1000,
                'max_goal_distance': 40000,
            }
        },
        {
            "name": "success",
            "class": DubinsSuccessStatus,
            "config": {
                "rejoin_time": "rejoin_time",
                'success_time': 20,
            }
        },
    ],
    "observation": [
        {
            "name": "observation_processor",
            "class": DubinsObservationProcessor,
            "config": {
                'lead': 'lead',
                'wingman': 'wingman',
                'rejoin_region': 'rejoin_region',
                'mode': 'magnorm',
                'reference': 'wingman',
            }
        }
    ],
    "reward": [
        {
            "name": "rejoin_reward",
            "class": RejoinRewardProcessor,
            "config": {
                'rejoin_status': "in_rejoin",
                'rejoin_prev_status': "in_rejoin_prev",
                'reward': 0.1,
            }
        },
        {
            "name": "rejoin_first_time_reward",
            "class": RejoinFirstTimeRewardProcessor,
            "config": {
                'rejoin_status': "in_rejoin",
                'reward': 0.25,
            }
        },
        {
            "name": "time_reward",
            "class": TimeRewardProcessor,
            "config": {
                'reward': -0.01,
            }
        },
        {
            "name": "rejoin_dist_change_reward",
            "class": RejoinDistanceChangeRewardProcessor,
            "config": {
                "wingman": "wingman",
                "rejoin_status": "in_rejoin",
                "rejoin_region": "rejoin_region",
                'reward': -0.00001,
            }
        },
        {
            "name": "failure_reward",
            "class": FailureRewardProcessor,
            "config": {
                "failure_status": "failure",
                "reward": {
                    'timeout': -1,
                    'crash': -1,
                    'distance': -1,
                }
            }
        },
        {
            "name": "success_reward",
            "class": SuccessRewardProcessor,
            "config": {
                "success_status": "success",
                'reward': 1,
            }
        },
    ],
    "verbose": False
}

config['env_config'] = rejoin_config
config['env'] = DubinsRejoin

stop_dict = {
    'training_iteration': 300,
}

### initial debugging
# env = DubinsRejoin(config=rejoin_config)

# env.reset()         # redundant?

# out1 = env.step((0, 0, 4))
# out2 = env.step((0, 0, 4))
# out3 = env.step((0, 0, 4))

### run with RLLib

def main(debug=False):

    # Create output dirs
    os.makedirs(output_dir, exist_ok=True)
    args_yaml_filepath = os.path.join(output_dir, 'script_args.yaml')
    ray_config_yaml_filepath = os.path.join(output_dir, 'ray_config.yaml')

    # Save experiment params
    with open(args_yaml_filepath, 'w') as args_yaml_file:
        arg_dict = vars(args)
        yaml.dump(arg_dict, args_yaml_file)
    with open(ray_config_yaml_filepath, 'w') as ray_config_yaml_file:
        yaml.dump(config, ray_config_yaml_file)

    if not debug:
        tune.run(ppo.PPOTrainer, config=config, stop=stop_dict, local_dir=args.output_dir, checkpoint_freq=25,
                 checkpoint_at_end=True, name=expr_name)
    else:
        # Run training in a single process for debugging
        config["num_workers"] = 0
        trainer = ppo.PPOTrainer(config=config)
        while True:
            print(trainer.train())


if __name__ == "__main__":
    # Workaround for YAML not dumping ABCMeta objects
    # TODO: See if there is a better way to fix this
    from yaml.representer import Representer
    from abc import ABCMeta
    Representer.add_representer(ABCMeta, Representer.represent_name)

    DEBUG = False

    main(debug=DEBUG)