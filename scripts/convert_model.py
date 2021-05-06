import numpy as np
import os

import pickle

import tensorflow as tf
import tensorflow.keras as keras

from scipy.io import savemat

import ray

import ray.rllib.agents.ppo as ppo

from saferl.aerospace.tasks.rejoin.task import DubinsRejoin
from saferl.environment.utils import numpy_to_matlab_txt

from contextlib import redirect_stdout
from collections import OrderedDict


tf.compat.v1.disable_eager_execution()

# expr_dir = 'output/expr_20210210_152136/PPO_DubinsRejoin_b926e_00000_0_2021-02-10_15-21-37'
# ckpt_num = 400

expr_dir = 'output/expr_20210210_164346/PPO_DubinsRejoin_3487a_00000_0_2021-02-10_16-43-49'
ckpt_num = 1000

ray_config_path = os.path.join(expr_dir, 'params.pkl')
ckpt_dir_name = 'checkpoint_{}'.format(ckpt_num)
ckpt_filename = 'checkpoint-{}'.format(ckpt_num)
ckpt_path = os.path.join(expr_dir, ckpt_dir_name, ckpt_filename)

with open(ray_config_path, 'rb') as ray_config_f:
    ray_config = pickle.load(ray_config_f)

ray.init()

env_config = ray_config['env_config']
ray_config['callbacks'] = ppo.DEFAULT_CONFIG['callbacks']

agent = ppo.PPOTrainer(config=ray_config, env=DubinsRejoin)
agent.restore(ckpt_path)

policy = agent.get_policy()
model = policy.model.base_model
weights = policy.get_weights()

obs = np.linspace(-.9, .9, 12)


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

env = DubinsRejoin(config=env_config)
env.seed(ray_config['seed'])

episode_obs = []
episode_logits = []
episode_actions = []
episode_value = []

# turn off explore
agent.get_policy().config['explore'] = False

obs = env.reset()
done = False
i = 0
while not done:
    print(i)
    i += 1
    logits, value = model_rollout.predict(obs[None, :])
    action = agent.compute_action(obs)

    episode_obs.append(obs)
    episode_logits.append(logits)
    episode_value.append(value)
    episode_actions.append(action)

    obs, reward, done, info = env.step(action)

logits_mat = np.array(episode_logits)[:, 0, :]
values_mat = np.array(episode_value)[:, 0, :]
obs_mat = np.array(episode_obs)

# save network inputs and outputs to npz file at mat file
model_test_io_npz_path = os.path.join(converted_ckpt_dir, 'ckpt_{:03d}_test_io.npz'.format(ckpt_num))
model_test_io_mat_path = os.path.join(converted_ckpt_dir, 'ckpt_{:03d}_test_io.mat'.format(ckpt_num))

np.savez(model_test_io_npz_path, logits=logits_mat, values=values_mat, obs=obs_mat)
savemat(model_test_io_mat_path, {'logits': logits_mat, 'values': values_mat, 'obs': obs_mat})


1 + 1
