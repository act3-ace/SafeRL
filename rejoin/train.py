import argparse
import gym
from gym.spaces import Discrete, Box
import numpy as np
import os

import pickle

import ray
from ray import tune

import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

from rejoin_rta.environments.rejoin_env import DubinsRejoin

config={
    'env': DubinsRejoin,
    'env_config':{},
}

stop = {
    "training_iteration": 50,
    "timesteps_total": 1000000,
}

def run_rollout(agent):
    # instantiate env class
    env = DubinsRejoin(agent)

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

# results = tune.run("PPO", config=config, stop=stop)

ray.init()
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
# config["eager"] = False

rollout_history = []



trainer = ppo.PPOTrainer(config=config, env=DubinsRejoin)

rollout_history.append(run_rollout(trainer))
for i in range(50):
    result = trainer.train()
    rollout_history.append(run_rollout(trainer))
    print(pretty_print(result))

pickle.dump( rollout_history, open( "save.pickle", "wb" ) )