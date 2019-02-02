# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from model import create_model
from DataGeneration import dataGeneration
from Checkpoint import createCheckpoint
from Performance import plot, calculate
from training import train
from Predict import predict

loadPreviouslySavedWeights = 1;

network = create_model()

dataSet = dataGeneration()

cp_callback = createCheckpoint()

train(network, dataSet, cp_callback, loadPreviouslySavedWeights)

#calculate(network, dataSet)

predict(network, dataSet)