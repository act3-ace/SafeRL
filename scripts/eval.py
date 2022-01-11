import os
import argparse
# import pickle5 as pickle
import pickle5 as pickle
import jsonlines
import tqdm
from glob import glob

import ray
import ray.rllib.agents.ppo as ppo

from saferl.environment.utils import jsonify, is_jsonable, YAMLParser, build_lookup
from saferl.environment.constants import RENDER

"""
This script loads an agent's policy from a saved checkpoint in the specified experiment directory. It randomly
initializes the environment said policy was trained on and runs the specified number of evaluation rollouts. These
evaluation rollout episodes are logged to a jsonlines eval.log file (found in experiment_dir/eval/chpt_<number> by
default). Currently, only DubinsRejoin and DockingEnv are supported.

Author: John McCarroll
"""

DEFAULT_TRIAL_INDEX = 0


class InvalidExperimentDirStructure(Exception):
    pass


def get_args():
    """
    A function to process script args.

    Returns
    -------
    argparse.Namespace
        Collection of command line arguments and their values
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default="", help="The full path to the experiment directory", required=True)
    parser.add_argument(
        '--trial_index',
        type=int,
        default=DEFAULT_TRIAL_INDEX,
        help="The index corresponding to the desired experiment to load. Use when multiple experiments are run by Tune."
        )
    parser.add_argument('--ckpt_num', type=int, default=None, help="Specify a checkpoint to load")
    parser.add_argument('--seed', type=int, default=None, help="The seed used to initialize evaluation environment")
    parser.add_argument('--explore', default=False, action="store_true", help="True for off-policy evaluation")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="The full path to the directory to write evaluation logs in")
    parser.add_argument('--num_rollouts', type=int, default=10,
                        help="Number of randomly initialized episodes to evaluate")
    parser.add_argument('--render', default=False, action="store_true", help="attempt to render evaluation episodes")
    parser.add_argument('--render_config', type=str, default=None,
                        help="The full path to a custom rendering config file")
    parser.add_argument('--alt_env_config', type=str, default="",
                        help="Provide the full path to an alternative environment config file,"
                             " in which the loaded policy will be evaluated.")

    return parser.parse_args()


def run_rollouts(agent, env, log_dir, num_rollouts=1, render=False):
    """
    A function to coordinate policy evaluation via RLLib API.

    Parameters
    ----------
    agent : ray.rllib.agents.trainer_template.PPO
        The trained agent which will be evaluated.
    env : BaseEnv
        The environment in which the agent will act.
    log_dir : str
        The path to the output directory in which evaluation logs will be run.
    num_rollouts : int
        The number of randomly initialized episodes conducted to evaluate the agent on.
    render : bool
        Flag to render the environment in a separate window during rollouts.
    """
    for i in tqdm.tqdm(range(num_rollouts)):
        # run until episode ends
        episode_reward = 0
        done = False
        obs = env.reset()
        step_num = 0

        with jsonlines.open(log_dir, "a") as writer:
            while not done:
                # progress environment state
                action = agent.compute_single_action(obs)
                obs, reward, done, info = env.step(action)
                step_num += 1
                episode_reward += reward

                # store log contents in state
                state = {}
                if is_jsonable(info) is True:
                    state["info"] = info
                else:
                    state["info"] = jsonify(info)
                state["actions"] = [float(i) for i in action]
                state["obs"] = obs.tolist()
                state["rollout_num"] = i
                state["step_number"] = step_num
                state["episode_reward"] = episode_reward

                # write state to file
                writer.write(state)

                if render:
                    # attempt to render environment state
                    env.render()


def verify_experiment_dir(expr_dir_path, trial_index=None):
    """
    A function to ensure passed path points to experiment run directory (as opposed to the parent directory).

    Parameters
    ----------
    expr_dir_path : str
        The full path to the experiment directory as received from arguments.
    trial_index : int
        The index of the trial directory to return, given experiment directory was the result of a Tune run with
        multiple trials.

    Returns
    -------
    The full path to the experiment run directory (containing the params.pkl file).
    """

    # look for params in given path
    params_file = "/params.pkl"
    given = glob(expr_dir_path + params_file)
    # look for params in child dirs
    children = sorted(glob(expr_dir_path + "/*" + params_file),
                      key=lambda trial_dir_path: int(trial_dir_path.split("/")[-2].split("_")[4]))

    if len(given) == 0:
        # params file not in given dir
        if len(children) == 0:
            raise InvalidExperimentDirStructure("No params.pkl file found!")
        elif len(children) > 1:
            # handle multiple episodes
            if trial_index:
                if trial_index < len(children) and trial_index >= -len(children):
                    file_name = children[trial_index]
                    expr_dir_path = file_name[:len(file_name) - len(params_file)]
                    return expr_dir_path
                else:
                    raise ValueError("No trial directory corresponding to index {} was found!"
                                     .format(trial_index))
            else:
                # take first trial in children
                size = len(children[0])
                expr_dir_path = children[0][:size - len(params_file)]
        else:
            # params file found in child dir
            size = len(children[0])
            expr_dir_path = children[0][:size - len(params_file)]

    return expr_dir_path


def find_checkpoint_dir(expr_dir_path, ckpt_num):
    """
    A function to locate the checkpoint and trailing identifying numbers of a checkpoint file.

    Parameters
    ----------
    ckpt_num : int
        The identifying checkpoint number which the user wishes to evaluate.

    Returns
    -------
    ckpt_num : int
        The specified ckpt number (or the number of the latest saved checkpoint, if no ckpt_num was specified).
    ckpt_num_str : str
        The trailing numbers of the checkpoint directory corresponding to the desired checkpoint.
    """
    ckpt_num_str = None

    # find checkpoint dir
    if ckpt_num is not None:
        ckpt_dirs = glob(expr_dir_path + "/checkpoint_*" + str(ckpt_num))
        if len(ckpt_dirs) == 1:
            ckpt_num_str = ckpt_dirs[0].split("_")[-1]
        else:
            raise FileNotFoundError("Checkpoint {} file not found".format(ckpt_num))

    else:
        ckpt_num = -1
        ckpt_dirs = glob(expr_dir_path + "/checkpoint_*")
        for ckpt_dir_name in ckpt_dirs:
            file_num = ckpt_dir_name.split("_")[-1]
            if ckpt_num < int(file_num):
                ckpt_num = int(file_num)
                ckpt_num_str = file_num

    return ckpt_num, ckpt_num_str


def parse_jsonlines_log(filepath):
    log_states = []
    with jsonlines.open(filepath, 'r') as log:
        for state in log:
            log_states.append(state)

    return log_states


def main():
    # process args
    args = get_args()

    # assume full path passed in
    expr_dir_path = args.dir
    trial_index = args.trial_index

    # verify experiment run dir
    expr_dir_path = verify_experiment_dir(expr_dir_path, trial_index=trial_index)

    # get checkpoint num
    ckpt_num, ckpt_num_str = find_checkpoint_dir(expr_dir_path, args.ckpt_num)

    # set paths
    eval_dir_path = os.path.join(expr_dir_path, 'eval')
    ckpt_eval_dir_path = os.path.join(eval_dir_path, 'ckpt_{}'.format(ckpt_num))

    ray_config_path = os.path.join(expr_dir_path, 'params.pkl')
    ckpt_dir = 'checkpoint_{}'.format(ckpt_num_str)
    ckpt_filename = 'checkpoint-{}'.format(ckpt_num)
    ckpt_path = os.path.join(expr_dir_path, ckpt_dir, ckpt_filename)

    # user specified output
    if args.output_dir is not None:
        eval_dir_path = args.output_dir
        ckpt_eval_dir_path = os.path.join(eval_dir_path, 'ckpt_{}'.format(ckpt_num))

    # make directories
    os.makedirs(eval_dir_path, exist_ok=True)
    os.makedirs(ckpt_eval_dir_path, exist_ok=True)

    # load checkpoint
    with open(ray_config_path, 'rb') as ray_config_f:
        ray_config = pickle.load(ray_config_f)

    # Load render config
    render_config = None
    if args.render_config is not None:
        parser = YAMLParser(yaml_file=args.render_config, lookup=build_lookup())
        render_config = parser.parse_env()

    ray.init()

    # load policy and env
    env_config = args.alt_env_config if args.alt_env_config else ray_config['env_config']
    if render_config is not None:
        env_config[RENDER] = render_config
    agent = ppo.PPOTrainer(config=ray_config, env=ray_config['env'])
    agent.restore(ckpt_path)
    env = ray_config['env'](env_config)

    # set seed and explore
    seed = args.seed if args.seed is not None else ray_config['seed']
    env.seed(seed)

    agent.get_policy().config['explore'] = args.explore

    # run inference episodes and log results
    run_rollouts(agent, env, ckpt_eval_dir_path + "/eval.log", num_rollouts=args.num_rollouts, render=args.render)


if __name__ == "__main__":
    main()
