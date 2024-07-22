from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


train_data_dir = "../../datasets/Classification/MIT_large_train/train"
test_data_dir = "../../datasets/Classification/MIT_large_train/test"
MODEL_FNAME = "best_model.h5"
#####
img_width = 299
img_height= 299
batch_size=10
number_of_epoch=50
validation_samples=807


# create the base pre-trained model
base_model = InceptionResNetV2(weights='imagenet')
#plot_model(base_model, to_file='modelInceptionResNetV2.png', show_shapes=True, show_layer_names=True)

# get the layer with name
layer = base_model.get_layer('block35_1')

# create a new model  that cuts the first one at the layer named before
model = Model(inputs=base_model.input, outputs=layer.output)
x = model.output

#Add an averagePooling to make it shape conpatible
x = GlobalAveragePooling2D()(x)

# add a new dense layer with the output we're looking for 
x = Dense(8, activation='softmax',name='predict1')(x)

# create a new model
model = Model(inputs=model.input, outputs=x)



plot_model(model, to_file='modelInceptionResNetV2_at_35_1.png', show_shapes=True, show_layer_names=True)
for layer in base_model.layers:
    layer.trainable = True
    
model.save(MODEL_FNAME)
