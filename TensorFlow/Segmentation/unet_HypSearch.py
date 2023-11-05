# -*- coding: utf-8 -*-


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
from tensorflow.keras import optimizers
import matplotlib
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice, plot_contour
from optuna.visualization import matplotlib as optuna_plots
#matplotlib.use('Agg')




epochs=3
val_samples=200

def get_optimizer(name, trial):
    if name == 'SGD':
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif name == 'Adadelta':
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    elif name == 'Adagrad':
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

    elif name == 'Nadam':
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1, log=True)
        return tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer name: {name}")
        
def plot_loss_accuracy_graphs(history, trial_number):
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_iou'], label="Training Mean IoU")
    plt.plot(history.history['val_mean_iou'], label="Validation Mean IoU")
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.title('Mean IoU vs. Epoch')
    plt.legend()

    # Save the plots as images
    subfolder = "results_hyper"
    os.makedirs(subfolder, exist_ok=True)
    output_filename = os.path.join(subfolder, f"loss_accuracy_trial_{trial_number}.png")
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()

def image_generator(files, batch_size = 32, sz = (256, 256)):

  while True:

    #extract a random batch
    batch = np.random.choice(files, size = batch_size)

    #variables for collecting batches of inputs and outputs
    batch_x = []
    batch_y = []


    for f in batch:

        #get the masks. Note that masks are png files
        mask = Image.open(f'annotations/{f[:-4]}.png')
        mask = np.array(mask.resize(sz))


        #preprocess the mask
        mask[mask >= 2] = 0
        mask[mask != 0 ] = 1

        batch_y.append(mask)

        #preprocess the raw images
        raw = Image.open(f'images/{f}')
        raw = raw.resize(sz)
        raw = np.array(raw)

        #check the number of channels because some of the images are RGBA or GRAY
        if len(raw.shape) == 2:
          raw = np.stack((raw,)*3, axis=-1)

        else:
          raw = raw[:,:,0:3]

        batch_x.append(raw)

    #preprocess a batch of images and masks
    batch_x = np.array(batch_x)/255.
    batch_y = np.array(batch_y)
    batch_y = np.expand_dims(batch_y,3)

    yield (batch_x, batch_y)

batch_size = 32

all_files = os.listdir('images')
shuffle(all_files)

split = int(0.95 * len(all_files))

#split into training and testing
train_files = all_files[0:split]
test_files  = all_files[split:]

train_generator = image_generator(train_files, batch_size = batch_size)
test_generator  = image_generator(test_files, batch_size = batch_size)





x, y= next(train_generator)

plt.axis('off')
img = x[0]
msk = y[0].squeeze()
msk = np.stack((msk,)*3, axis=-1)

plt.imshow( np.concatenate([img, msk, img*msk], axis = 1))

"""# IoU metric

The intersection over union (IoU) metric is a simple metric used to evaluate the performance of a segmentation algorithm. Given two masks $y_{true}, y_{pred}$ we evaluate

$$IoU = \frac{y_{true} \cap y_{pred}}{y_{true} \cup y_{pred}}$$
"""

def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou

"""# Model"""

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

"""# Callbacks

Simple functions to save the model at each epoch and show some predictions
"""

def build_callbacks():
        checkpointer = ModelCheckpoint(filepath='unet.h5', verbose=0, save_best_only=True)
        callbacks = [checkpointer, PlotLearning()]
        return callbacks

# inheritance for training process plot
class PlotLearning(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('mean_iou'))
        self.val_acc.append(logs.get('val_mean_iou'))
        self.i += 1
        print('i=',self.i,'loss=',logs.get('loss'),'val_loss=',logs.get('val_loss'),'mean_iou=',logs.get('mean_iou'),'val_mean_iou=',logs.get('val_mean_iou'))
        # Save the plot to a file (change the filename and format as needed)
        # Create a subfolder for saving the plot if it doesn't exist
        subfolder = "results/trials"
        os.makedirs(subfolder, exist_ok=True)
        
        # Save the plot to a file inside the subfolder (change the filename and format as needed)
        output_filename = os.path.join(subfolder, "training_"+str(self.i)+".png") 
        #choose a random test image and preprocess
        path = np.random.choice(test_files)
        raw = Image.open(f'images/{path}')
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
        plt.savefig(output_filename, bbox_inches="tight")
        plt.show()
        #########################
        # Save loss and accuracy plots as images
        plt.figure(figsize=(12, 6))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.x, self.losses, label="Training Loss")
        plt.plot(self.x, self.val_losses, label="Validation Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs. Epoch')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.x, self.acc, label="Training Mean IoU")
        plt.plot(self.x, self.val_acc, label="Validation Mean IoU")
        plt.xlabel('Epoch')
        plt.ylabel('Mean IoU')
        plt.title('Mean IoU vs. Epoch')
        plt.legend()
        
        # Save the plots as images
        output_filename = os.path.join(subfolder, "loss__t"+".png")
        plt.savefig(output_filename, bbox_inches="tight")
        plt.close()
        # Increment the epoch counter
        #self.i += 1
        ########################






"""# Training"""



def objective(trial):
    optimizer_name = trial.suggest_categorical('optimizer', ['SGD',  'Adagrad', 'Adadelta', 'Nadam'])
    batchSize = trial.suggest_int('batchSize', 32, 64, step=16)
    opt= get_optimizer(optimizer_name, trial)
    model = unet()

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[mean_iou])

    validation_samples = val_samples
    train_steps = len(train_files) //batchSize
    test_steps =  int(validation_samples //batchSize)+1
     
    history=model.fit_generator(train_generator,
                        epochs = epochs, steps_per_epoch=train_steps ,validation_data = test_generator, validation_steps = test_steps,
                        callbacks = build_callbacks(), verbose = 0)
    val_loss = history.history['val_loss'][-1]
    plot_loss_accuracy_graphs(history, trial.number)
    
    return val_loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)
best_params = study.best_params
print("Best Hyperparameters_unet:", best_params)
# Get the trials DataFrame
trials_df = study.trials_dataframe()

# Save the DataFrame to a .csv file
trials_df.to_csv('optuna_results_unet.csv', index=False)

# Save the DataFrame to an .xlsx file
trials_df.to_excel('optuna_results_unet.xlsx', index=False)
plot = plot_optimization_history(study)
plot.write_image("optimization_history_unet.png")
plot1 = plot_param_importances(study)
plot1.write_image("param_unet.png")
plot2 = plot_slice(study)
plot2.write_image("slice_unet.png")
plot3 = plot_contour(study)
plot3.write_image("contour_unet.png")



"""# Testing"""

# #!wget http://r.ddmcdn.com/s_f/o_1/cx_462/cy_245/cw_1349/ch_1349/w_720/APL/uploads/2015/06/caturday-shutterstock_149320799.jpg -O test.jpg
# for i in range(0,7):
#     raw = Image.open('test/'+str(i)+'.jpg')
#     raw = np.array(raw.resize((256, 256)))/255.
#     raw = raw[:,:,0:3]
    
#     #predict the mask
#     pred = model.predict(np.expand_dims(raw, 0))
    
#     #mask post-processing
#     msk  = pred.squeeze()
#     msk = np.stack((msk,)*3, axis=-1)
#     msk[msk >= 0.5] = 1
#     msk[msk < 0.5] = 0
    
#     #show the mask and the segmented image
#     combined = np.concatenate([raw, msk, raw* msk], axis = 1)
#     plt.axis('off')
#     #plt.imshow(combined)
#     # Save the plot to a file (change the filename and format as needed)
#     # Create a subfolder for saving the plot if it doesn't exist
#     subfolder = "results"
#     os.makedirs(subfolder, exist_ok=True)
    
#     # Save the plot to a file inside the subfolder (change the filename and format as needed)
#     output_filename = os.path.join(subfolder, "test_"+str(i)+".jpg") 
#     plt.show()
#     plt.savefig(output_filename, bbox_inches="tight")






