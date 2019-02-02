# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:19:47 2019

@author: pradeep
"""

from keras.preprocessing.image import ImageDataGenerator

def dataGeneration():
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range = 15)
    test_datagen = ImageDataGenerator(rescale = 1./255)    
    training_set = train_datagen.flow_from_directory('data/training_data', 
                                                     target_size=(128,128),
                                                     color_mode='rgb',
                                                     batch_size = 2,
                                                     class_mode = 'binary'
                                                     )    
    valid_set = test_datagen.flow_from_directory('data/validation_data',
                                                  target_size = (128,128),
                                                  color_mode = 'rgb',
                                                  batch_size = 2,
                                                  class_mode = 'binary'
                                                  )    
    test_set = test_datagen.flow_from_directory('data/test_data',
                                                target_size = (128,128),
                                                class_mode = 'binary',
                                                shuffle = False,
                                                batch_size = 1
                                                )
    return [training_set, valid_set, test_set]
