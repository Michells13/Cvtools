from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LayerNormalization, Dropout

import re

def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

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

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory()
            #if insert_layer_name:
            #    new_layer.name = insert_layer_name
            #else:
            #    new_layer.name = '{}_{}'.format(layer.name, 
            #                                    new_layer.name)
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                            layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names:
            model_outputs.append(x)

    return Model(inputs=model.inputs, outputs=model_outputs)


init_model = "./models/35_1_not_trained.h5"

# Model
model = load_model(init_model)

# Replace batch normalization by layer normalization
#model = insert_layer_nonseq(model, "batch_norm.*", LayerNormalization, position="replace")

#model.save('workingModel.h5')
#model = load_model('workingModel.h5')

#plot_model(model, to_file='modelLayerNormalization.png', show_shapes=True, show_layer_names=True)

# Add Dropout after each convolution
def returnDropout():
    return Dropout(0.5)
model = insert_layer_nonseq(model, ".*global.*", returnDropout, position="after")

model.save('workingModel.h5')
model = load_model('workingModel.h5')

plot_model(model, to_file='modelDropout.png', show_shapes=True, show_layer_names=True)


print("Number of parameters: ",  model.count_params())
model.save('initialRemoved.h5')

# Change activation
def changeActivation(model, activation):
    model = insert_layer_nonseq(model, ".*ac.*", activation, position="replace")
    model.save('workingModel.h5')
    model = load_model('workingModel.h5')
    return model