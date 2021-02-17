# pylint: disable=no-member

from typing import Dict, Optional
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.policy.sample_batch import SampleBatch
import pickle
import os
import json
import jsonlines


def build_callbacks_caller(callbacks : [] ):

    class CallbacksCaller(DefaultCallbacks):
        def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
            self.callbacks = callbacks
            super(CallbacksCaller, self).__init__(legacy_callbacks_dict)

        def on_episode_end(self, *args, **kwargs):
            for callback in self.callbacks:
                if callable(getattr(callback, "on_episode_end", None)):
                    callback.on_episode_end(*args, **kwargs)
            super(CallbacksCaller, self).on_episode_end(*args, **kwargs)

        def on_episode_step(self, *args, **kwargs):
            for callback in self.callbacks:
                if callable(getattr(callback, "on_episode_step", None)):
                    callback.on_episode_step(*args, **kwargs)
            super(CallbacksCaller, self).on_episode_step(*args, **kwargs)

        def on_postprocess_trajectory(self, *args, **kwargs):
            for callback in self.callbacks:
                if callable(getattr(callback, "on_postprocess_trajectory", None)):
                    callback.on_postprocess_trajectory(*args, **kwargs)
            super(CallbacksCaller, self).on_postprocess_trajectory(*args, **kwargs)

    return CallbacksCaller
    

class EpisodeOutcomeCallback:
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        episode.custom_metrics["outcome/success"] = int(episode.last_info_for()['success'])
        episode.custom_metrics["outcome/failure"] = int(bool(episode.last_info_for()['failure']))


class FailureCodeCallback:
    def __init__(self, failure_codes = ['timeout', 'distance', 'crash']):
        self.failure_codes = failure_codes

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        if episode.last_info_for()['failure']:
            for failure_code in self.failure_codes:
                episode.custom_metrics["failure_code_ratio/{}".format(failure_code)] = int(episode.last_info_for()['failure'] == failure_code)

class RewardComponentsCallback:
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        
        ep_info = episode.last_info_for()
        for reward_comp_name, reward_comp_val in ep_info['reward']['component_totals'].items():
            episode.custom_metrics['reward_component_totals/{}'.format(reward_comp_name)] = reward_comp_val

"""
A callback class to handle the storage of episode states by episode
"""
class LoggingCallback:
    def __init__(self, num_logging_workers: int = 999999, episodes_omitted_before_log: int = 0):
        self.num_logging_workers = num_logging_workers
        self.episodes_omitted_before_log = episodes_omitted_before_log
        self.episodes = set()

    def on_episode_step(self, *, worker: "RolloutWorker", base_env: BaseEnv, episode: MultiAgentEpisode,
                        env_index: Optional[int] = None, **kwargs) -> None:

        ## Debug
        # if episode.episode_id in self.phone_book:
        #     self.phone_book[episode.episode_id] = self.phone_book[episode.episode_id].append(episode.length)
        # else:
        #     self.phone_book[episode.episode_id] = [episode.length]
        # print(self.phone_book)
        # print(self)
        # print(self.phone_book[episode.episode_id])

        # get environment instance and set up log path
        episode_id = episode.episode_id
        worker_index = worker.worker_index
        output_dir = worker._original_kwargs["log_dir"] + "../training_logs/"
        worker_file = "worker_" + str(worker_index) + ".log"
        step_num = episode.length

        # handle logging options
        self.episodes.add(episode_id)
        if worker_index <= self.num_logging_workers and len(self.episodes) % self.episodes_omitted_before_log == 0:
            state = {}
            state["actions"] = episode.last_action_for('agent0').tolist()       # TODO: 'agent0' should not be hardcoded...*
            state["obs"] = episode.last_raw_obs_for('agent0').tolist()
            state["info"] = episode.last_info_for('agent0')
            state["episode_ID"] = episode_id
            state["step_num"] = step_num

            # save environment state to file
            self.log_to_file(state, output_dir, worker_file)

    # Helper function to handle writing to file
    def log_to_file(self, state, output_dir, jsonline_filename):
        os.makedirs(output_dir, exist_ok=True)
        with jsonlines.open(output_dir + jsonline_filename, mode='a') as writer:
            writer.write(state)
