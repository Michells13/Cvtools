from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Add
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx
def remove_layers(model, layer_finish, layer_start, layer_ind_finish, layer_ind_start):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    network_dict['input_layers_of'][layer_start] = [layer_finish]

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    layerIndex = 0
    for layer in model.layers[1:]:

        layerIndex += 1

        if layerIndex > layer_ind_finish and layerIndex < layer_ind_start:
            continue
        
        
        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1 or layerIndex == len(model.layers) - 1:
            layer_input = layer_input[0]

        x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})
        
        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names:
            model_outputs.append(x)
    return Model(inputs=model.inputs, outputs=model_outputs)

#train_data_dir='/ghome/mcv/datasets/MIT_split/train'
train_data_dir = "../W3/MIT_split/train/"
#val_data_dir='/ghome/mcv/datasets/MIT_split/test'
test_data_dir = "../W3/MIT_split/test"
#test_data_dir='/ghome/mcv/datasets/MIT_split/test'

img_width = 299
img_height= 299
batch_size=32
number_of_epoch=20
validation_samples=807

NUM_BLOCKS_REM = 32

# create the base pre-trained model
base_model = InceptionResNetV2(weights='imagenet')
plot_model(base_model, to_file='modelInceptionResNetV2.png', show_shapes=True, show_layer_names=True)

# Modify last layer
x = base_model.layers[-2].output
x = Dense(8, activation='softmax',name='predictions')(x)

model = Model(inputs=base_model.input, outputs=x)

if NUM_BLOCKS_REM != 0:

    bigGroupRemove = NUM_BLOCKS_REM//2
    smallGroupRemove = bigGroupRemove//2
    
    # Remove from first block group
    layer_finish = "block35_" + str(10-smallGroupRemove)
    layer_finish_index = getLayerIndexByName(model, layer_finish)
    layer_start = "block35_10_ac"
    layer_start_index = getLayerIndexByName(model, layer_start)
    model = remove_layers(model, layer_finish, layer_start, layer_finish_index, layer_start_index)
    
    model.save('workingModel.h5')
    model = load_model('workingModel.h5')
    
    # Remove from second block group
    layer_finish = "block17_" + str(20-bigGroupRemove)
    layer_finish_index = getLayerIndexByName(model, layer_finish)
    layer_start = "block17_20_ac"
    layer_start_index = getLayerIndexByName(model, layer_start)
    model = remove_layers(model, layer_finish, layer_start, layer_finish_index, layer_start_index)
    
    model.save('workingModel.h5')
    model = load_model('workingModel.h5')
    
    # Remove from second block group
    layer_finish = "block8_" + str(10-smallGroupRemove)
    layer_finish_index = getLayerIndexByName(model, layer_finish)
    layer_start = "conv_7b"
    layer_start_index = getLayerIndexByName(model, layer_start)
    model = remove_layers(model, layer_finish, layer_start, layer_finish_index, layer_start_index)
    
    model.save('workingModel.h5')
    model = load_model('workingModel.h5')

    plot_model(model, to_file='modelInceptionResNetV2removed.png', show_shapes=True, show_layer_names=True)


print("Number of parameters: ",  model.count_params())
model.save('initialRemoved.h5')