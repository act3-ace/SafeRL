import argparse
import os
import numpy as np

from tensorflow import keras

import onnx
import tf2onnx

parser = argparse.ArgumentParser()

# Add parser arguments
parser.add_argument('ckpt', type=str)
args = parser.parse_args()

ckpt_path_sans_ext = os.path.splitext(args.ckpt)[0]
output_path_keras = ckpt_path_sans_ext + "_with_pre_post_processing" + "_keras.h5"
output_path_onnx = ckpt_path_sans_ext + "_with_pre_post_processing" + ".onnx"

model_weights_set_dict = {}

model_orig = keras.models.load_model(args.ckpt)

input_size = 4
action_size = 2

input_layer = keras.Input(shape=(input_size,))

normalized_output = keras.layers.Dense(input_size, name="preprocess_norm_scale")(input_layer)
model_weights_set_dict['preprocess_norm_scale'] = [
    np.diag([1/100, 1/100, 2, 2]),
    np.zeros(input_size),
]

orig_policy_output, _ = model_orig(normalized_output)

# ouput post processing
policy_means_clipped = keras.layers.Dense(action_size, name="postprocess_filter_std_clip", activation="tanh"
                                          )(orig_policy_output)
model_weights_set_dict['postprocess_filter_std_clip'] = [
    np.diag([1, 0, 1, 0])[::2, :].T,
    np.zeros(action_size),
]


output_model = keras.Model(inputs=input_layer, outputs=policy_means_clipped)

# set weights
for layer_name, layer_weights in model_weights_set_dict.items():
    output_model.get_layer(name=layer_name).set_weights(layer_weights)

output_model.save(output_path_keras)

model_proto, external_tensor_storage = tf2onnx.convert.from_keras(output_model)

onnx.save(model_proto, output_path_onnx)

# preprocessing layers
