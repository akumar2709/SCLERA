#example of how to convert a tensorflow model to flite


import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


classifier = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

input_data = np.array(np.random.rand(1,224,224,3), dtype=np.float32)



converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.target_spec.supported_types = [tf.float32]
tflite_model = converter.convert()

# Save the model.
with open('tflite_model.tflite', 'wb') as f:
  f.write(tflite_model)
