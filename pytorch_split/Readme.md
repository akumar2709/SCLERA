This script splits resnet50 and saves it in onnx and pt format for demo purposes. The script could be modified to work with other resnet models.
Onnx-Tensorflow library(https://github.com/onnx/onnx-tensorflow.git) could be used to convert the given model to tensorflow proto buffer format 
To convert the ProtoBuffer to tflite, you could use the script provided in the tensorflow_split folder
