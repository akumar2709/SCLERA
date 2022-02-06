import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
resnet101 = "https://tfhub.dev/google/bit/s-r101x1/ilsvrc2012_classification/1"
IMAGE_SHAPE = (224, 224)

#model = ResNet101(weights='imagenet')

"""classifier = tf.keras.Sequential([
    hub.KerasLayer(resnet101, input_shape=IMAGE_SHAPE+(3,))
])"""
"""classifier = tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000)
h5_model = "./mymodel.h5"
#model = hub.load("./enformer_1")"""
classifier = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

"""classifier = tf.keras.Sequential([
    hub.KerasLayer("./enformer_1",input_shape=IMAGE_SHAPE+(3,))
])"""
#model = hub.load("./enformer_1").model
#tf.saved_model.save(model, h5_model)
input_data = np.array(np.random.rand(1,224,224,3), dtype=np.float32)
#print(model.input)

#converter = tf.lite.TFLiteConverter.from_saved_model("./wrn28_4_teacher")
converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS # enable TensorFlow Lite ops.
  #tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.target_spec.supported_types = [tf.float32]
tflite_model = converter.convert()
tflite_quant_model = converter.convert()
open("vgg16_32.tflite", "wb").write(tflite_quant_model)
