"""
Computer Vision Final Project
Deepfake Classification Model
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from tensorflow.keras.layers import \
	Conv2D, MaxPool2D, Dropout, Flatten, Dense
class DeepFakeModel(tf.keras.Model):
    '''
    hyperparameters here
    '''
    def __init__(self):
        super(DeepFakeModel, self).__init__()
    
        # optimizer:

        # architexture
        self.architecture = []
