import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from tensorflow import keras

#imports pretrained model
model = ResNet50(weights='imagenet')


#split the model, using the model and layer number as split point
def model_split(model, layer_number):
#Use the layer number to get the name of the layer that is the point of split
    layer_name = "conv" + str(layer_number+2) + str("_block1_1_conv")
    input_layer = keras.Input(shape=model.get_layer(layer_name).input.shape[1:])
#getting the layer index in the model of the point of split 
    layer_index = 0
    for layer in model.layers:
        if layer.name == layer_name:
            break
        layer_index += 1
    x = input_layer
    prev_out = input_layer

    model1 = keras.Model(inputs = model.input, outputs = model.layers[layer_index-1].output)
#using layer index to loop through the model and recreate the second half of the model
    for i in range(layer_index, len(model.layers)):
        if(len(model.layers[i].name) <= 11):
            x = model.layers[i](x)
        elif(model.layers[i].name[13] == '0' and model.layers[i].name[15:] == "conv"):
            x1 = model.layers[i](prev_out)
            x1 = model.layers[i+2](x1)
        elif(model.layers[i].name[13] != '0' and model.layers[i].name[13:] != "add"):
            x = model.layers[i](x)
            if(model.layers[i].name[13:] == "out"):
                prev_out = x
        elif(model.layers[i].name[13:] == "add"):
            if(model.layers[i].name[11] == '1'):
                x = keras.layers.Add()([x,x1])
            else:
                x = keras.layers.Add()([x, prev_out])
           
    model2 = keras.Model(inputs = input_layer, outputs = x)
#returning both the generated models as a objects in a list
    return [model1, model2]

model_split = model_split(model, 2)

#saving the models in tflite format
split_index = 1
for split_model in model_split:
    converter = tf.lite.TFLiteConverter.from_keras_model(split_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS # enable TensorFlow Lite ops.
   
    ]
    
    #converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    tflite_quant_model = converter.convert()
    model_name = "resnet50_" + str(split_index) + ".tflite"
    open(model_name, "wb").write(tflite_quant_model)
    split_index += 1
