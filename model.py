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


# from IPython import get_ipython
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Flatten, Activation
from keras.models import Sequential, Model
from keras import applications

os.environ["CUDA_VISIBLE_DEVICES"]="1"

img_width, img_height = 420, 420

# ## Autoencoder with VGG
# Firstly we load vgg network with wieghts trained on huge data set
# Important: we setup input shape of model with image size as in previus network

base_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# We created proxy to converty output of vgg (13,13,512) -> into supported input of previsly trained network (416,416,3)

proxy_model= Sequential()
proxy_model.add(Conv2D(3, (1, 1), activation='relu', padding='same',  input_shape=base_model.output_shape[1:]))
proxy_model.add(UpSampling2D((32, 32)))
proxy_model.add(ZeroPadding2D((1, 1)))

# Combine networks. Important how we setup inputs and outputs

base_model_and_proxy = Model(inputs=base_model.input, outputs=proxy_model(base_model.output))

# Recreate model and load weights of previusly trined network
# base_model_and_proxy  (418,418,3)

top_model = Sequential()
top_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=base_model_and_proxy.output_shape[1:]))
top_model.add(MaxPooling2D((2, 2), padding='same'))
top_model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
top_model.add( MaxPooling2D((2, 2), padding='same'))
top_model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
encoded=MaxPooling2D((2, 2), padding='same')
top_model.add(encoded)


top_model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
top_model.add(UpSampling2D((2, 2)))
top_model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
top_model.add(UpSampling2D((2, 2)))
top_model.add(Conv2D(16, (3, 3), activation='relu'))
top_model.add(UpSampling2D((2, 2)))
decoded=Conv2D(3, (3, 3), activation='sigmoid', padding='same')
top_model.add(decoded)
# top_model.load_weights('first_try-20epoch.h5')

# Combine networks

autoencoder = Model(inputs=base_model_and_proxy.input, outputs=top_model(base_model_and_proxy.output))


# Make sure that we don`t train  vgg network

for layer in autoencoder.layers[:19]:
    layer.trainable = True

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#autoencoder.compile(loss='mean_squared_error', optimizer = 'adam')