# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:42:19 2024

@author: MICHE
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from keras.layers import  Dropout, Activation
from keras.optimizers import Adam, SGD
from keras.layers import ELU, PReLU, LeakyReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.utils import plot_model
import tensorflow as tf
import glob
import random
import cv2
from random import shuffle

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

"""# Unet Model"""

def unet(sz = (256, 256, 3)):
  x = Input(sz)
  inputs = x

  #down sampling
  f = 8
  layers = []

  for i in range(0, 6):
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    layers.append(x)
    x = MaxPooling2D() (x)
    f = f*2
  ff2 = 64

  #bottleneck
  j = len(layers) - 1
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
  x = Concatenate(axis=3)([x, layers[j]])
  j = j -1

  #upsampling
  for i in range(0, 5):
    ff2 = ff2//2
    f = f // 2
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j -1


  #classification
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  outputs = Conv2D(1, 1, activation='sigmoid') (x)

  #model creation
  model = Model(inputs=[inputs], outputs=[outputs])
  #model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = [mean_iou])

  return model

model = unet()
model = unet()
model.load_weights('unet_simpleTraining.h5')
folder='test_images/'



"""# Testing"""

#!wget http://r.ddmcdn.com/s_f/o_1/cx_462/cy_245/cw_1349/ch_1349/w_720/APL/uploads/2015/06/caturday-shutterstock_149320799.jpg -O test.jpg
for i in range(0,7):
    raw = Image.open(folder+str(i)+'.jpeg')
    raw = np.array(raw.resize((256, 256)))/255.
    raw = raw[:,:,0:3]
    
    #predict the mask
    pred = model.predict(np.expand_dims(raw, 0))
    
    #mask post-processing
    msk  = pred.squeeze()
    msk = np.stack((msk,)*3, axis=-1)
    msk[msk >= 0.5] = 1
    msk[msk < 0.5] = 0
    
    #show the mask and the segmented image
    combined = np.concatenate([raw, msk, raw* msk], axis = 1)
    plt.axis('off')
    plt.imshow(combined)
    # Save the plot to a file (change the filename and format as needed)
    # Create a subfolder for saving the plot if it doesn't exist
    subfolder = "results"
    os.makedirs(subfolder, exist_ok=True)
    
    # Save the plot to a file inside the subfolder (change the filename and format as needed)
    output_filename = os.path.join(subfolder, folder +str(i)+".jpeg") 
    plt.show()
    plt.savefig(output_filename, bbox_inches="tight")