###coded by sno....

#### New image generation flow
import cv2
import numpy as np
import pandas as pd
import os, fnmatch

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
print(os.getcwd())


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

        
def data(initial_image_dir,train_data_dir,validation_data_dir):
    image_list = os.listdir(initial_image_dir) #initial path to images
    print(initial_image_dir)
    inital_image_count=0
    for img in image_list:   
        img_path= initial_image_dir + '/' + img
        if not os.path.isfile(img_path):
            continue
        
        inital_image_count += 1  
        
        img = load_img(img_path)  # this is a PIL image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        
        train_save_to = train_data_dir + '/correct_samples'
        # print train_save_to
        if not os.path.exists(train_save_to):
            os.makedirs(train_save_to)
        
        valid_save_to = validation_data_dir + '/correct_samples'
        if not os.path.exists(valid_save_to):
            os.makedirs(valid_save_to)
        
        print("generate additional images for train in: " + train_save_to)
        
        i = 0
        for batch in datagen.flow(x, batch_size=5, save_to_dir = train_save_to, save_prefix='sample', save_format='jpeg'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely

        train_size=0 
        # print train_size
        for t in os.listdir(train_save_to):
            if os.path.isfile(train_save_to + '/' + t):
                train_size += 1 
                
                
        print("generate additional images for validation in: " + valid_save_to)
        ii=0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=valid_save_to, save_prefix='doc', save_format='jpeg'):
            ii += 1
            if ii > 4:
                break  # otherwise the generator would loop indefinitely
                
        validation_size=0           
        for v in os.listdir(valid_save_to):
            if os.path.isfile(valid_save_to + '/' + v):
                validation_size += 1 
    return inital_image_count,train_size,validation_size
# print("-------------------------------------------")
# print("Initial image count: {} ".format(inital_image_count))
# print("Train image count: {} ".format(train_size))
# print("Validation image count: {} ".format(validation_size))