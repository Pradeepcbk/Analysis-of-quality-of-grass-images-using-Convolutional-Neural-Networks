# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:17:46 2019

@author: pradeep
"""
from keras.models import Sequential
from keras import optimizers
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization

def create_model():
    model = Sequential()
    adam = optimizers.Adam(lr = 0.001, decay = 0.0, beta_1 = 0.9, beta_2 = 0.999, epsilon = None, amsgrad = False)
    
    model.add(Conv2D(8, (3,3), input_shape = (128,128,3), activation  = 'relu'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size = (2,2)))   
    
    model.add(Conv2D(16, (3,3), activation  = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Conv2D(32, (3,3), activation  = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    
    model.add(Flatten())
    
    model.add(Dense(units = 196, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(Dense(units = 1, activation='sigmoid', kernel_initializer='he_normal', bias_initializer='zeros'))
    
    model.compile(optimizer = adam, loss = 'binary_crossentropy',metrics = ['accuracy'])
    return model
