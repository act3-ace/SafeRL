import argparse
import numpy as np
import os

import pickle

import tensorflow as tf
import tensorflow.keras as keras

from scipy.io import savemat

import ray

# deprecated,  see https://discuss.ray.io/t/ray-rllib-agents-ppo-missing/9904
#import ray.rllib.agents.ppo as ppo

#from ray import rllib
#from ray.rllib.algorithms.ppo import PPO as ppo

from ray.rllib.algorithms.ppo import PPO

#from ray.rllib import algorithms
#from ray.rllib.algorithms import ppo
#from ray.rllib.algorithms.ppo import PPO as ppo



from saferl.environment.utils import numpy_to_matlab_txt

from contextlib import redirect_stdout
from collections import OrderedDict

tf.compat.v1.disable_eager_execution()

#<<<<<<< HEAD
#expr_dir = "output/expr_20220316_102941/PPO_DockingEnv_8625c_00000_0_2022-03-16_10-29-44"
#=======
#expr_dir = "output/expr_20220316_102941/PPO_DockingEnv_8625c_00000_0_2022-03-16_10-29-44"i
#>>>>>>> 5f00c09693ef5b9a62bec3789c57ba657ab9a749
#expr_dir = "output/expr_20230720_124921/PPO_DockingEnv_884c9_00000_0_2023-07-20_12-49-26"
expr_dir = "output/expr_20230731_110254/PPO_DockingEnv_80031_00000_0_2023-07-31_11-03-07"

#ckpt_num = 35
ckpt_num = 200

parser = argparse.ArgumentParser()

# Add parser arguments
parser.add_argument('expr_dir', type=str)
parser.add_argument('ckpt_num', type=int)
args = parser.parse_args()

expr_dir = args.expr_dir
ckpt_num = args.ckpt_num

ray_config_path = os.path.join(expr_dir, 'params.pkl')
ckpt_dir_name = 'checkpoint_{:06d}'.format(ckpt_num)
ckpt_filename = 'checkpoint-{}'.format(ckpt_num)
ckpt_path = os.path.join(expr_dir, ckpt_dir_name, ckpt_filename)

with open(ray_config_path, 'rb') as ray_config_f:
    ray_config = pickle.load(ray_config_f)

ray.init()

env_config = ray_config['env_config']
ray_config['callbacks'] = ppo.DEFAULT_CONFIG['callbacks']

agent = ppo.PPOTrainer(config=ray_config, env=ray_config['env'])
agent.restore(ckpt_path)

policy = agent.get_policy()
model = policy.model.base_model
weights = policy.get_weights()

sess = policy.get_session()
tf.compat.v1.keras.backend.set_session(sess)

for layer in model.layers:
    if hasattr(layer, 'kernel_initializer'):
        layer.kernel_initializer = keras.initializers.Constant()

converted_ckpt_dir = os.path.join(expr_dir, 'converted_ckpt')
os.makedirs(converted_ckpt_dir, exist_ok=True)
keras_save_path = os.path.join(converted_ckpt_dir, 'ckpt_{:03d}.h5'.format(ckpt_num))
model.save(keras_save_path)

# save model summary
model_summary_save_path = os.path.join(converted_ckpt_dir, 'ckpt_{:03d}_model_summary.txt'.format(ckpt_num))
with open(model_summary_save_path, 'w') as f:
    with redirect_stdout(f):
        model.summary()

# create new weight dict with weight names / --> _
weights_formatted = OrderedDict()
for name, mat in weights.items():
    weights_formatted[name.replace("/", "_")] = mat

# save weights as matlab txt
mat_txt_save_path = os.path.join(converted_ckpt_dir, 'ckpt_{:03d}.txt'.format(ckpt_num))
with open(mat_txt_save_path, 'w') as mat_txt_f:
    for weight_name, weight_mat in weights_formatted.items():
        numpy_to_matlab_txt(weight_mat, name=weight_name, output_stream=mat_txt_f)

# save weights as matlab code
mat_code_save_path = os.path.join(converted_ckpt_dir, 'ckpt_{:03d}.m'.format(ckpt_num))
with open(mat_code_save_path, 'w') as mat_code_f:
    for weight_name, weight_mat in weights_formatted.items():
        numpy_to_matlab_txt(weight_mat, name=weight_name, output_stream=mat_code_f)

# save weights to mat file
mat_save_path = os.path.join(converted_ckpt_dir, 'ckpt_{:03d}.mat'.format(ckpt_num))
savemat(mat_save_path, weights_formatted)

# run model through an environment episode and save observations/model output
model_rollout = tf.keras.models.load_model(keras_save_path)

env = ray_config['env'](env_config)
env.seed(ray_config['seed'])

# turn off explore
agent.get_policy().config['explore'] = False

trials = []
for trial_idx in range(10):
    episode_data = {
        'obs': [],
        'info': [],
        'policy': [],
        'value': [],
        'action': [],
        'control': [],
    }
    obs = env.reset()

    info = {}
    for obj in env.env_objs.items():
        info[obj[0]] = obj[1].generate_info()


#    info = {
#        'wingman': env.env_objs['wingman']._generate_info(),
#        'lead': env.env_objs['lead']._generate_info(),
#        'rejoin_region': env.env_objs['rejoin_region']._generate_info(),
#    }

    episode_data['obs'].append(obs)
    episode_data['info'].append(info)
    episode_data['policy'].append([])
    episode_data['value'].append([])
    episode_data['action'].append([])
    episode_data['control'].append([])

    done = False

    i = 0
    while not done:
        # print(i)
        i += 1
        policy, value = model_rollout.predict(obs[None, :])
        action = agent.compute_single_action(obs)
        # print("RLLIB:")
        # print(action)

        # action_output = np.clip(policy, -1, 1)
        # action = (action_output[0, 0],)
        action = tuple(np.split(policy[0, ::2], policy.shape[1]/2))
        # print("KERAS:")
        # print(action)

        obs, reward, done, info = env.step(action)

        control = np.copy(env.env_objs['deputy'].current_control)

        episode_data['obs'].append(obs)
        episode_data['info'].append(info)
        episode_data['policy'].append(policy)
        episode_data['value'].append(value)
        episode_data['action'].append(action)
        episode_data['control'].append(control)

    episode_data['policy_mat'] = np.array(episode_data['policy'][1:])[:, 0, :]
    episode_data['values_mat'] = np.array(episode_data['value'][1:])[:, 0, :]
    episode_data['obs_mat'] = np.array(episode_data['obs'][:-1])

    if episode_data['info'][-1]['success']:
        outcome = 'success'
    else:
        outcome = 'failure: ' + episode_data['info'][-1]['failure']

    episode_data['outcome'] = outcome

    trials.append(episode_data)

    print(i, info['success'], info['failure'])

# save network inputs and outputs to npz file at mat file
model_test_io_npz_path = os.path.join(converted_ckpt_dir, 'ckpt_{:03d}_test_io.npz'.format(ckpt_num))
model_test_io_pkl_path = os.path.join(converted_ckpt_dir, 'ckpt_{:03d}_test_io.pickle'.format(ckpt_num))
model_test_io_mat_path = os.path.join(converted_ckpt_dir, 'ckpt_{:03d}_test_io.mat'.format(ckpt_num))

np.savez(model_test_io_npz_path, trials=trials)
# savemat(model_test_io_mat_path, {'trials': trials})
with open(model_test_io_pkl_path, 'wb') as f:
    pickle.dump({'trials': trials}, f)
