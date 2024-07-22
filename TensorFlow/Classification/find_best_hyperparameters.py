from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, linear
from replace_activation_layers import changeActivation
from tensorflow.keras.utils import plot_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna
import os
import numpy as np


#train_data_dir='/ghome/mcv/datasets/MIT_split/train'
train_data_dir = "./MIT_small_train_1/train/"
#val_data_dir='/ghome/mcv/datasets/MIT_split/test'
test_data_dir = "./MIT_small_train_1/test"
#test_data_dir='/ghome/mcv/datasets/MIT_split/test'
validation_samples=2288
img_width = 299
img_height= 299
MODEL_FNAME = "model_trained.h5"
initial_model = "./models/35_1_not_trained.h5"
folderHyper = "./findHypers/"

if not os.path.exists(folderHyper):
    os.makedirs(folderHyper)

# Hyperparameter values
batch_sizes = [8, 16, 32, 64, 128]
number_of_epochs = [10, 50, 100]
optimizersIndex = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
optimizers = {"SGD": SGD, "RMSprop": RMSprop, "Adagrad": Adagrad, 
              "Adadelta": Adadelta, "Adam": Adam, "Adamax": Adamax, 
              "Nadam": Nadam}
learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
momentums = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
activationsIndex = ["softmax", "softplus", "softsign", "relu", "tanh", 
                    "sigmoid", "hard_sigmoid", "Linear"]
activations = {"softmax": softmax, "softplus": softplus, 
               "softsign": softsign, "relu": relu, "tanh": tanh, 
               "sigmoid": sigmoid, "hard_sigmoid": hard_sigmoid, 
               "Linear": linear}
data_augmentations = {"width_shift_range": [0.1, 0], "height_shift_range": [0.1, 0], 
                      "horizontal_flip": [True, False], "vertical_flip": [True, False],
                      "rotation_range": [10,0], "brightness_range": [True, False],
                      "zoom_range": [0.1, 0], "shear_range": [10, 0]}



def train_network(trial):
    
    # create the base pre-trained model
    model = load_model(initial_model)
    # Choose hyperparameters
    batch_size = trial.suggest_categorical("batch_size", batch_sizes)
    number_of_epoch = trial.suggest_categorical("epochs", number_of_epochs)
    optimizerIndex = trial.suggest_categorical("optimizer", optimizersIndex)
    optimizer = optimizers[optimizerIndex]
    learning_rate = trial.suggest_categorical("learning_rate", learning_rates)
    if optimizerIndex == "SGD" or optimizerIndex == "RMSprop":
        momentum = trial.suggest_categorical("momentum", momentums)
        opt = optimizer(learning_rate = learning_rate, momentum = momentum)
    else:
        opt = optimizer(learning_rate = learning_rate)
    activationIndex = trial.suggest_categorical("activation", activationsIndex)
    activation = activations[activationIndex]
    model = changeActivation(model, activation)
    
    width_shift_range = trial.suggest_categorical("width_shift_range", data_augmentations["width_shift_range"])
    height_shift_range = trial.suggest_categorical("height_shift_range", data_augmentations["height_shift_range"])
    horizontal_flip = trial.suggest_categorical("horizontal_flip", data_augmentations["horizontal_flip"])
    vertical_flip = trial.suggest_categorical("vertical_flip", data_augmentations["vertical_flip"])
    rotation_range = trial.suggest_categorical("rotation_range", data_augmentations["rotation_range"])
    brightness_range = trial.suggest_categorical("brightness_range", data_augmentations["brightness_range"])
    if brightness_range:
        brightness_range = [0.8,1.2]
    else:
        brightness_range = [1.,1.]
    zoom_range = trial.suggest_categorical("zoom_range", data_augmentations["zoom_range"])
    shear_range = trial.suggest_categorical("shear_range", data_augmentations["shear_range"])
        
    
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    model.summary()

    datagenTrain = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
    	preprocessing_function=preprocess_input,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        brightness_range= brightness_range,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=None)
    
    datagenTest = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
    	preprocessing_function=preprocess_input,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None)
    
    train_generator = datagenTrain.flow_from_directory(train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')
    
    test_generator = datagenTest.flow_from_directory(test_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')


    # To save the best model
    checkpointer = ModelCheckpoint(filepath=MODEL_FNAME, verbose=1, save_best_only=True, 
                                   monitor='val_accuracy')
    
    history=model.fit(train_generator,
            steps_per_epoch=(int(np.ceil(400/batch_size))),
            epochs=number_of_epoch,
            validation_data=test_generator,
            validation_steps= (int(np.ceil(validation_samples/batch_size))), callbacks=[checkpointer])
    
    model = load_model(MODEL_FNAME)
    
    result = model.evaluate(test_generator)
    
    newFolder = folderHyper + "trial" + str(trial.number) + "/"
    if not os.path.exists(newFolder):
        os.makedirs(newFolder)
    model.save(newFolder + MODEL_FNAME)
    # Save results
    file  = open(newFolder + "results.txt", "w")
    file.write(f"Score: {result}\n")
    file.write(f"Batch size: {batch_size}\n")
    file.write(f"Epochs: {number_of_epoch}\n")
    file.write(f"Optimizer: {optimizerIndex}\n")
    file.write(f"Lr: {learning_rate}\n")
    if optimizerIndex == "SGD" or optimizerIndex == "RMSprop":
        file.write(f"Momentum: {momentum}\n")
    file.write(f"Activation: {activationIndex}\n")
    file.write("DATA AUGMENTATION: \n")
    file.write(f"width_shift_range: {width_shift_range}\n")
    file.write(f"height_shift_range: {height_shift_range}\n")
    file.write(f"horizontal_flip: {horizontal_flip}\n")
    file.write(f"vertical_flip: {vertical_flip}\n")
    file.write(f"rotation_range: {rotation_range}\n")
    file.write(f"brightness_range: {brightness_range}\n")
    file.write(f"zoom_range: {zoom_range}\n")
    file.write(f"shear_range: {shear_range}\n")
    file.close()
    
    fig1, ax1 = plt.subplots()
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'validation'], loc='upper left')
    fig1.savefig(newFolder + 'accuracy.jpg')
    plt.close(fig1)
      # summarize history for loss
    fig1, ax1 = plt.subplots()
    ax1.plot(history.history['loss'])
    ax1.plot(history.history['val_loss'])
    ax1.set_title('model loss')
    ax1.set_ylabel('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(['train', 'validation'], loc='upper left')
    fig1.savefig(newFolder + 'loss.jpg')
    
    plot_model(model, to_file=newFolder + 'modelInceptionResNetV2changed.png', show_shapes=True, show_layer_names=True)

    return result[1]

# Create Study object
study = optuna.create_study(direction="maximize")
# Optimize the study
study.optimize(train_network, n_trials=50) # Use more 
# Print the result
best_params = study.best_params
best_score = study.best_value
print(f"Best score: {best_score}\n")
print(f"Optimized parameters: {best_params}\n")

# plots
fig = optuna.visualization.plot_slice(study, params=['batch_size', 'epochs', 'optimizer', 
                                                     'learning_rate', "momentum", "activation"])
fig.write_image("plot_slice.png")
fig = optuna.visualization.plot_slice(study, params=['width_shift_range', 'height_shift_range', 'horizontal_flip', 
                                                     'vertical_flip', "rotation_range", "brightness_range",
                                                     "zoom_range", "shear_range"])
fig.write_image("plot_slice2.png")
fig = optuna.visualization.plot_contour(study, params=['batch_size', 'epochs', 'optimizer', 
                                                     'learning_rate', "momentum", "activation"])
fig.write_image("plot_contour.png")
fig = optuna.visualization.plot_contour(study, params=['width_shift_range', 'height_shift_range', 'horizontal_flip', 
                                                     'vertical_flip', "rotation_range", "brightness_range",
                                                     "zoom_range", "shear_range"])
fig.write_image("plot_contour2.png")
# Save results
file  = open("results.txt", "w")
file.write(f"Best score: {best_score}\n")
file.write(f"Optimized parameters: {best_params}\n")
file.close()