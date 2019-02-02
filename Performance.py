# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:34:17 2019

@author: pradeep
"""
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import tensorflow as tf

def calculate(network, dataSet):    
    predictions = network.predict_generator(dataSet[2], steps = 680)
    predicted_classes = predictions > 0.5
    
    true_classes = dataSet[2].classes
    class_labels = list(dataSet[2].class_indices.keys())
    
    report = classification_report(true_classes, predicted_classes, target_names = class_labels)
    
    print(report)
    
    confusionMatrix = confusion_matrix(y_true = true_classes, y_pred = predicted_classes)
    
    print(confusionMatrix)
    
def plot(network):
    tf.keras.utils.plot_model(
        network,
        to_file='model.png',
        show_shapes=True,
        show_layer_names=False
    )
    
    