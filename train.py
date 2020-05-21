###coded by sno....

# ### Load libs
import os
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Activation
from keras.models import Sequential, Model
from keras import applications
from  model import autoencoder
from data_aug import data

os.environ["CUDA_VISIBLE_DEVICES"]="1"

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

# ### Helpers. Params. Preprocesing

def fixed_generator(generator):
    for batch in generator:
        yield (batch, batch)



img_width, img_height = 420, 420

batch_size = 2 #32

nb_validation_samples=0
nb_train_samples=0

nb_epoch=5

initial_image_dir='images/docs'
train_data_dir = initial_image_dir + '/train'
validation_data_dir = initial_image_dir + '/valid'

# #### Generator for images to complete dataset
# Generator is used for extending the image dataset by image transformation

datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# #### New image generation flow
# #### To enable data augmentation uncomment below line

# inital_image_count,train_size,validation_size = data(initial_image_dir,train_data_dir,validation_data_dir)           
# print("-------------------------------------------")
# print("Initial image count: {} ".format(inital_image_count))
# print("Train image count: {} ".format(train_size))
# print("Validation image count: {} ".format(validation_size))

# this is the augmentation configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures
train_generator = train_datagen.flow_from_directory(
        train_data_dir,  # this is the target directory
        target_size=(img_width, img_height),  
        batch_size=batch_size,
        color_mode='rgb', 
        class_mode=None)  

nb_train_samples=train_generator.samples
# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        color_mode='rgb', 
        class_mode=None)

nb_validation_samples=validation_generator.samples

# Training

autoencoder.fit_generator(
        fixed_generator(train_generator),
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=nb_epoch,
        validation_data=fixed_generator(validation_generator),
        validation_steps=nb_validation_samples // batch_size)

autoencoder.save_weights('autoencoder-vgg.h5')

