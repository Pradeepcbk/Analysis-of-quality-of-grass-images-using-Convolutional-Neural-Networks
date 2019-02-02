# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:45:32 2019

@author: pradeep
"""

def train(network, dataSet, cp_callback, load):
    if load:
        network.load_weights('checkpoints/cp-0010.ckpt')

    else:
        network.fit_generator(dataSet[0],
                          steps_per_epoch = 800/2,  # steps_per_epoch * batch_size = dataset_count
                          epochs = 10,
                          validation_steps = 104/2,
                          callbacks = [cp_callback],
                          validation_data = dataSet[1],
                          verbose=1)