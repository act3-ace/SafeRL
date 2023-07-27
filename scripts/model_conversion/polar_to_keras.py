'''
polar_to_keras.py takes in model checkpoints and converts to onnx format.
@created 20230727
@modified 20230727
@auth ieva
'''

import tensorflow as tf
import tf2onnx
#/home/ieja/SafeRL/output
#\\wsl$\Ubuntu\home\ieja\SafeRL\output
checkpoint_path = r"C:\Users\USER\Desktop\docking_local\clbf_controller\epoch=50-step=458.ckpt"
output_path = "path/to/output/model.onnx"

graph_def = tf.GraphDef()

with tf.io.gfile.GFile(checkpoint_path, "rb") as f:
    graph_def.ParseFromString(f.read())

onnx_model, _ = tf2onnx.convert.from_graphdef(graph_def, input_names=["input"], output_names=["output"])
tf2onnx.save_model(onnx_model, output_path)



