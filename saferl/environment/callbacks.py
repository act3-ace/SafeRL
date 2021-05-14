# pylint: disable=no-member

from typing import Dict, Optional
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from saferl.environment.utils import jsonify, is_jsonable, log_to_jsonlines

import os
import time
import numpy as np
import json
import jsonlines
from enum import Enum


def build_callbacks_caller(callbacks: []):
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
    def __init__(self, failure_codes=['timeout', 'distance', 'crash']):
        self.failure_codes = failure_codes

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        if episode.last_info_for()['failure']:
            for failure_code in self.failure_codes:
                episode.custom_metrics["failure_code_ratio/{}".format(failure_code)] = int(
                    episode.last_info_for()['failure'] == failure_code)


class RewardComponentsCallback:
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        ep_info = episode.last_info_for()
        for reward_comp_name, reward_comp_val in ep_info['reward']['components']['total'].items():
            episode.custom_metrics['reward_component_totals/{}'.format(reward_comp_name)] = reward_comp_val


class LogContents(Enum):
    """
    Simple Enum class for log contents options
    """

    INFO = "info"
    ACTIONS = "actions"
    OBS = "obs"
    VERBOSE = "verbose"


class LoggingCallback:
    """
    A callback class to handle the storage of episode states by episode
    """

    def __init__(self, num_logging_workers: int = 999999, episode_log_interval: int = 1,
                 contents: tuple = (LogContents.VERBOSE,)):
        self.num_logging_workers = num_logging_workers
        self.episode_log_interval = episode_log_interval

        self.worker_episode_numbers = dict()
        self.episode_count = 0

        self.log_actions = False
        self.log_obs = False
        self.log_info = False

        for content in contents:
            if content == LogContents.VERBOSE:
                self.log_actions = True
                self.log_obs = True
                self.log_info = True
                break
            if content == LogContents.INFO:
                self.log_info = True
            if content == LogContents.OBS:
                self.log_obs = True
            if content == LogContents.ACTIONS:
                self.log_actions = True

    def on_episode_step(self, *, worker: "RolloutWorker", base_env: BaseEnv, episode: MultiAgentEpisode,
                        env_index: Optional[int] = None, **kwargs) -> None:

        if episode.episode_id not in self.worker_episode_numbers:
            self.worker_episode_numbers[episode.episode_id] = self.episode_count
            self.episode_count += 1

        # get environment instance and set up log path
        episode_id = episode.episode_id
        worker_index = worker.worker_index

        # determine output location
        output_dir = None
        if worker.policy_config["in_evaluation"]:
            output_dir = worker._original_kwargs["log_dir"] + "../evaluation_logs/"
        else:
            output_dir = worker._original_kwargs["log_dir"] + "../training_logs/"

        worker_file = "worker_" + str(worker_index) + ".log"
        step_num = episode.length

        # handle logging options
        if worker_index <= self.num_logging_workers \
                and self.worker_episode_numbers[episode_id] % self.episode_log_interval == 0 \
                and step_num:

            state = {}
            if self.log_actions:
                state["actions"] = episode.last_action_for('agent0').tolist()  # TODO: 'agent0' should not be hardcoded
            if self.log_obs:
                state["obs"] = episode.last_raw_obs_for('agent0').tolist()
            if self.log_info:
                # check if jsonable and convert if necessary
                info = episode.last_info_for('agent0')

                if is_jsonable(info) == True:
                    state["info"] = info
                else:
                    state["info"] = jsonify(info)

            state["episode_ID"] = episode_id
            state["step_number"] = step_num
            state["worker_episode_number"] = self.worker_episode_numbers[episode_id]
            state["time"] = time.time()

            # save environment state to file
            log_to_jsonlines(state, output_dir, worker_file)
