from typing import Dict, Optional
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.policy.sample_batch import SampleBatch
import pickle
import os

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

    def on_episode_step(self, *, worker: "RolloutWorker", base_env: BaseEnv, episode: MultiAgentEpisode,
                        env_index: Optional[int] = None, **kwargs) -> None:

        # print(worker.__dict__)
        # print(base_env.__dict__)
        # print(episode.__dict__)
        # print(env_index)
        # print(episode.length)

        # get environment instance and set up log path
        env = base_env.get_unwrapped()[env_index]
        episode_id = episode.episode_id
        output_dir = worker._original_kwargs["log_dir"]
        episode_dir = "DubinsTest-ep" + str(episode_id) + "/"
        step_num = episode.length

        state = {}
        state["info_lead"] = env.env_objs["lead"]._generate_info()
        state["info_wingman"] = env.env_objs["wingman"]._generate_info()
        state["info_rejoin_region"] = env.env_objs["rejoin_region"]._generate_info()
        # state["obs"] = env._generate_obs()                        # causes Attribute Error, despite being defined in rejoin_env.py
        state["actions"] = episode.last_action_for('agent0')       # Should not be hardcoded...***
        state["obs"] = episode.last_raw_obs_for('agent0')
        state["info"] = episode.last_info_for('agent0')
        state["episode_ID"] = episode_id

        # save environment state to file
        self.log_to_file(state, output_dir + episode_dir, step_num)

    # Helper function to handle writing to file
    def log_to_file(self, state, episode_dir_path, step_num):
        os.makedirs(episode_dir_path, exist_ok=True)
        with open(episode_dir_path + "step_" + str(step_num) + ".log", 'wb') as file:
            pickle.dump(state, file)


# Backlog
## "finalize" log contents
## remove hardcoded agentID