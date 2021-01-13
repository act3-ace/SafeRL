import unittest
from unittest import mock
import math
import numpy as np
import random
import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
import gym
from gym.spaces import Discrete, Box

from typing import Dict, Optional
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.policy.sample_batch import SampleBatch

from ray.tune.error import TuneError
import pickle
import os
import glob
import sys
sys.path.append("..")
from ..rejoin.rejoin_rta.aero_models import dubins
from ..rejoin.rejoin_rta.utils import util
from ..rejoin.rejoin_rta.utils import geometry
from ..rejoin.rejoin_rta.environments.rejoin_env import DubinsRejoin



class RandomCallback(DefaultCallbacks):
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: MultiAgentEpisode, env_index: Optional[int] = None, **kwargs) -> None:
        # get Dubins environment instance
        env = base_env.get_unwrapped()[env_index]

        # save sample of numpy rng to central queue
        path = os.path.abspath(os.path.curdir) + "/have-deepsky/tests/test_data/" + str(os.getpid())
        with open(path, 'wb') as file:
            pickle.dump(env._sample_random(5), file)

        # end the rllib experiment
        class TestOverException(Exception):
            pass
        raise TestOverException


class TestDubinsRejoin(unittest.TestCase):
    def setUp(self):
        self.reward_config = {
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
        self.rejoin_config = {
            'reward': self.reward_config,
            'init': {
                'wingman': {
                    'x': [-4000, 4000],
                    'y': [-4000, 4000],
                    'theta': [0, 2 * math.pi],
                    'velocity': [10, 100]
                },
                'lead': {
                    'x': [-4000, 4000],
                    'y': [-4000, 4000],
                    'theta': [0, 2 * math.pi],
                    'velocity': [40, 60]
                },
            },
            'obs': {
                # 'mode': 'rect',
                # 'reference': 'global',
                'mode': 'polar',
                'reference': 'wingman',
            },
            'rejoin_region': {
                'type': 'circle',
                'range': 500,
                'aspect_angle': 60,
                'radius': 150,
            },
            'constraints': {
                'safety_margin': {
                    'aircraft': 100
                },
                'timeout': 1000,
                'max_goal_distance': 40000,
                'success': {
                    'rejoin_time': 20,
                },
            },
            'verbose': False,
        }
        self.env = DubinsRejoin(self.rejoin_config)
        self.array_length = 5

    def test_env_obj(self):
        self.assertEqual(type(self.env.env_objs["wingman"]), dubins.DubinsAgent)
        self.assertEqual(type(self.env.env_objs["lead"]), dubins.DubinsAircraft)
        self.assertEqual(type(self.env.env_objs['rejoin_region']), geometry.RelativeCircle2D)

    # TODO: mocking dependencies to test simple delegating functions
    def test_obs_space(self):
        self.assertTrue(True)

    def test_action_space(self):
        self.assertTrue(True)

    def test_reset(self):
        self.assertTrue(True)

    def test_step(self):
        self.assertTrue(True)


    # Seed tests
    def test_seed_return(self):
        seed = 219
        self.assertEqual(self.env.seed(seed), [seed])

    def test_seed(self):
        # seed numpy locally
        seed = 514
        np.random.seed(seed)
        random_array = np.random.randn(self.array_length)

        # seed numpy via env
        self.env.seed(seed)
        env_random_array = self.env._sample_random(self.array_length)

        # compare resulting random arrays
        if np.array_equal(random_array, env_random_array):
            self.assertTrue(True)
        else:
            raise self.fail("Inconsistent elements returned from seeded numpy.random.randn()")

    def test_seed_rllib(self):
        # setup and init RLlib experiment
        ray.init(num_gpus=0)
        config = ppo.DEFAULT_CONFIG.copy()
        config["num_gpus"] = 0
        config["num_workers"] = 6
        config['_fake_gpus'] = True
        config['seed'] = 0
        config['callbacks'] = RandomCallback
        config['env_config'] = self.rejoin_config
        config['env'] = DubinsRejoin
        stop_dict = {
            'training_iteration': 200,
        }

        # clear test_data directory
        test_data_dir = os.path.abspath(os.path.curdir) + "/have-deepsky/tests/test_data/"
        os.makedirs(test_data_dir, exist_ok=True)
        files = glob.glob(test_data_dir + "*")
        for f in files:
            os.remove(f)

        # initiate the experiment via friendly Tune
        with self.assertRaises(TuneError):
            tune.run(ppo.PPOTrainer, config=config, stop=stop_dict, local_dir=None, checkpoint_freq=25,
                 checkpoint_at_end=True, name='rng_test')


        results = list()
        files = os.listdir(test_data_dir)
        for file_name in files:
            with open(test_data_dir + file_name, 'rb') as f:
                results.append(pickle.load(f))

        # check each saved rng result for consistency
        reference = results[0]
        for array in results:
            if not np.array_equal(reference, array):
                self.fail("Inconsistent elements returned from numpy.randn() from environments used by RLlib rollout workers")

        self.assertTrue(True, "Consistent numpy RNG results from RLlib rollout workers' environments")


if __name__ == '__main__':
    unittest.main()


# TODO:
# investigate way to remove verbosity of Tune/RLlib test - crowds unittest results display
# implement tests for environment's other methods