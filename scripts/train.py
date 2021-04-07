import argparse
import os

import yaml
import math

from datetime import datetime

import ray
from ray import tune

import ray.rllib.agents.ppo as ppo

from saferl.environment import build_callbacks_caller, EpisodeOutcomeCallback, FailureCodeCallback, \
                                        RewardComponentsCallback, LoggingCallback, LogContents, RelativeCircle

from saferl.aerospace.tasks import *
from saferl.aerospace.models import *


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, default='./output')

args = parser.parse_args()

expr_name =  datetime.now().strftime("expr_%Y%m%d_%H%M%S")
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

rollout_history = []


# ----------- New config style ----------------

# --------------- Rejoin ----------------------

rejoin_config = {
    "env_objs": [
        {
            "name": "wingman",
            "class": Dubins2dPlatform,
            "config": {
                "controller": {
                    "actuators": [
                        {
                            'name': 'rudder',
                            'space': 'discrete',
                            'points': 5,
                        },
                        {
                            'name': 'throttle',
                            'space': 'discrete',
                            'points': 5,
                        },
                    ]
                },
                "init": {
                    'x': [-4000, 4000],
                    'y': [-4000, 4000],
                    'heading': [0, 2*math.pi],
                    'v': [10, 100]
                }
            },
        },
        {
            "name": "lead",
            "class": Dubins2dPlatform,
            "config": {
                "init": {
                    'x': [-4000, 4000],
                    'y': [-4000, 4000],
                    'heading': [0, 2*math.pi],
                    'v': [40, 60]
                }
            },

        },
        {
            "name": "rejoin_region",
            "class": RelativeCircle,
            "config": {
                'ref': 'lead',
                'track_orientation': True,
                'r_offset': 500,
                'aspect_angle': 60,
                'radius': 150,
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


# --------------- Docking ----------------------

docking_config = {
    "env_objs": [
        {
            "name": "deputy",
            "class": CWHSpacecraft2d,
            "config": {
                'controller': {
                    'actuators': [
                        {
                            'name': 'thrust_x',
                            'space': 'discrete',
                            'points': 11,
                            'bounds': [-10, 10]
                        },
                        {
                            'name': 'thrust_y',
                            'space': 'discrete',
                            'points': 11,
                            'bounds': [-10, 10]
                        },
                    ]
                },
                "init": {
                    'x': [-2000, 2000],
                    'y': [-2000, 2000],
                    'x_dot': 0,
                    'y_dot': 0,
                }
            },
        },
        {
            "name": "chief",
            "class": CWHSpacecraft2d,
            "config": {
                "init": {
                    'x': 0,
                    'y': 0,
                    'x_dot': 0,
                    'y_dot': 0,
                }
            },

        },
        {
            "name": "docking_region",
            "class": RelativeCircle,
            "config": {
                'ref': 'chief',
                'x_offset': 0,
                'y_offset': 0,
                'radius': 20,
                "init": {}
            },
        },
    ],
    "agent": "deputy",
    "status": [
        {
            "name": "docking_status",
            "class": DockingStatusProcessor,
            "config": {
                "docking_region": "docking_region",
                "deputy": "deputy"
            }
        },
        {
            "name": "docking_distance",
            "class": DockingDistanceStatusProcessor,
            "config": {
                "docking_region": "docking_region",
                "deputy": "deputy"
            }
        },
        {
            "name": "failure",
            "class": FailureStatusProcessor,
            "config": {
                'timeout': 1000,
                "docking_distance": "docking_distance",
                "max_goal_distance": 40000
            }
        },
        {
            "name": "success",
            "class": SuccessStatusProcessor,
            "config": {
                "docking_status": "docking_status",
            }
        },
    ],
    "observation": [
        {
            "name": "observation_processor",
            "class": DockingObservationProcessor,
            "config": {
                'deputy': 'deputy',
                'mode': '2d',
            }
        }
    ],
    "reward": [
        {
            "name": "time_reward",
            "class": TimeRewardProcessor,
            "config": {
                'reward': -0.01,
            }
        },
        {
            "name": "dist_change_reward",
            "class": DistanceChangeRewardProcessor,
            "config": {
                "deputy": "deputy",
                "docking_region": "docking_region",
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


# --------------- Register environment ----------------------

config['env_config'] = rejoin_config
config['env'] = DubinsRejoin

stop_dict = {
    'training_iteration': 200,
}


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

    DEBUG = True

    main(debug=DEBUG)
