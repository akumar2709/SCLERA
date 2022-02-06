import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow import keras
classifier = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)
#model.summary()

def model_split(model, layer_name):
  input_layer  = keras.Input(shape=model.get_layer(layer_name).input.shape[1:])
  layer_index = 0 
  
  for layer in model.layers:
    if layer.name == layer_name:
      break
    layer_index += 1
  model1 = keras.Model(inputs = model.input, outputs = model.layers[layer_index-1].output)
  x = input_layer
  prev_out = input_layer
  for i in range(layer_index, len(model.layers)):
    x = model.layers[i](x)
  model2 = keras.Model(inputs=input_layer, outputs = x)
  return [model1, model2]

model_split = model_split(classifier, "block3_conv1")
split_index = 1
for split_model in model_split:
    converter = tf.lite.TFLiteConverter.from_keras_model(split_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS # enable TensorFlow Lite ops.
    #tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    #converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    tflite_quant_model = converter.convert()
    model_name = "vgg16_" + str(split_index) + ".tflite"
    open(model_name, "wb").write(tflite_quant_model)
    split_index += 1
