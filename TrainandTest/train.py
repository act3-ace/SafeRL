import argparse
import os

import yaml
import math

from datetime import datetime

import ray
from ray import tune

import ray.rllib.agents.ppo as ppo

from aerospaceSafeRL.environment import build_callbacks_caller, EpisodeOutcomeCallback, FailureCodeCallback, \
                                        RewardComponentsCallback, LoggingCallback, LogContents

from aerospaceSafeRL.AerospaceTasks import *


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

# ------ Define reward configuration ------

reward_processors = [
    RejoinRewardProcessor,
    RejoinFirstTimeRewardProcessor,
    TimeRewardProcessor,
    RejoinDistanceChangeRewardProcessor,
    FailureRewardProcessor,
    SuccessRewardProcessor
]

reward_config = {
    'processors': reward_processors,
    'time_decay': -0.01,
    'failure': {
        'timeout': -1,
        'crash': -1,
        'distance': -1,
    },
    'success': 1,
    'rejoin_timestep': 0.1,
    'rejoin_first_time': 0.25,
    'dist_change': -0.00001,
}

# ------ Define status configuration ------

status_processors = [
    DubinsInRejoin,
    DubinsInRejoinPrev,
    DubinsRejoinTime,
    DubinsTimeElapsed,
    DubinsLeadDistance,
    DubinsFailureStatus,
    DubinsSuccessStatus
]

status_config = {
    'processors': status_processors,
    'safety_margin': {
        'aircraft': 100
    },
    'timeout': 1000,
    'max_goal_distance': 40000,
    'success': {
        'rejoin_time': 20,
    },
}

# ------ Define observation configuration ------

observation_processors = [
    DubinsObservationProcessor
]

observation_config = {
    'processors': observation_processors,
    # 'mode': 'rect',
    # 'reference': 'global',
    'mode': 'magnorm',
    'reference': 'wingman',
}

rejoin_config = {
    'reward': reward_config,
    'init': {
        'wingman': {
            'x': [-4000, 4000],
            'y': [-4000, 4000],
            'heading': [0, 2*math.pi],
            'v': [10, 100]
        },
        'lead': {
            'x': [-4000, 4000],
            'y': [-4000, 4000],
            'heading': [0, 2*math.pi],
            'v': [40, 60]
        },
    },
    'agent':{
        'controller':{
            'actuators': [
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
            ],
        },
    },
    'observation': observation_config,
    'rejoin_region': {
        'type': 'circle',
        'range': 500,
        'aspect_angle': 60,
        'radius': 150,
    },
    'status': status_config,
    'verbose': False,
}

config['env_config'] = rejoin_config
config['env'] = DubinsRejoin

stop_dict = {
    'training_iteration': 200,
}


if __name__ == "__main__":
    # Workaround for YAML not dumping ABCMeta objects
    # TODO: See if there is a better way to fix this
    from yaml.representer import Representer
    from abc import ABCMeta
    Representer.add_representer(ABCMeta, Representer.represent_name)

    DEBUG = False

    # create output dir and save experiment params
    os.makedirs(output_dir, exist_ok=True)
    args_yaml_filepath = os.path.join(output_dir, 'script_args.yaml')
    ray_config_yaml_filepath = os.path.join(output_dir, 'ray_config.yaml')
    with open(args_yaml_filepath, 'w') as args_yaml_file:
        arg_dict = vars(args)
        yaml.dump(arg_dict, args_yaml_file)
    with open(ray_config_yaml_filepath, 'w') as ray_config_yaml_file:
        yaml.dump(config, ray_config_yaml_file)

    if not DEBUG:
        tune.run(ppo.PPOTrainer, config=config, stop=stop_dict, local_dir=args.output_dir, checkpoint_freq=25, checkpoint_at_end=True, name=expr_name)
    else:
        # Run training in a single process for debugging
        config["num_workers"] = 0
        trainer = ppo.PPOTrainer(config=config)
        while True:
            print(trainer.train())
