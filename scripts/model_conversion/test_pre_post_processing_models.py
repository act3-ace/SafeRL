import argparse
from multiprocessing.sharedctypes import Value
import os
import numpy as np

import pickle

from tensorflow import keras

import onnx
import onnxruntime as ort
import tf2onnx

parser = argparse.ArgumentParser()

# Add parser arguments
parser.add_argument('ckpt', type=str)
parser.add_argument('trail_io_data_path', type=str)
args = parser.parse_args()

ckpt_ext = os.path.splitext(args.ckpt)[1]

if ckpt_ext == '.h5':
    model_type = 'keras'
    model = keras.models.load_model(args.ckpt)
elif ckpt_ext == '.onnx':
    model_type = 'onnx'
    sess = ort.InferenceSession(args.ckpt)
else:
    raise ValueError("unrecognized ckpt format")

with open(args.trail_io_data_path, 'rb') as f:
    io_data = pickle.load(f)

trials = io_data['trials']

max_error = 0
for trial in trials:
    model_control = 0
    for info in trial['info']:
        state = info['deputy']['state'][None, :]
        control_actual = info['deputy']['controller']['control']

        error = np.linalg.norm(control_actual - model_control)
        max_error = max(max_error, error)

        # compute model control for next timestep
        if model_type == 'keras':
            model_control = model.predict(state)[0, :]
        else:
            model_control = sess.run(None, {'input_1': state.astype(np.float32)})[0][0, :]


print(f"max error = {max_error}")
