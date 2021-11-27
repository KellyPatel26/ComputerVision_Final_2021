import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, Sequential, Input, Model, applications
#os.environ['CUDA_VISIBLE_DEVICES'] = ""


class CNNBlock(layers.Layer):
    
    def __init__(self, name, h, w, size, pooling):
        super(CNNBlock, self).__init__()
        self.resnet = applications.ResNet50V2(include_top=False, input_shape=(h, w, 3))
        self.h = h
        self.w = w
        self.size = size
        self.pooling = pooling
        self.dh = h
        self.dw = w
        _, self.dh, self.dw, _ = self.resnet.output_shape

    def call(self, x):
        x = tf.reshape(x, [-1, self.h, self.w, 3])
        x = self.resnet(x)
        if self.pooling:
            x = layers.GlobalAveragePooling2D()(x)
            x = tf.reshape(x, [-1, self.size, 1, 1, 2048])
        else:
            x = tf.reshape(x, [-1, self.size, self.dh, self.dw, 2048])
        return x


class LSTMBlock(layers.Layer):
    
    def __init__(self, name, size, pooling):
        super(LSTMBlock, self).__init__()
        self.size = size
        self.pooling = pooling
        self.model = Sequential([
            layers.ConvLSTM2D(filters=512,
                              kernel_size=(3, 3),
                              padding="same",
                              strides=(1, 1), 
                              return_sequences=True,
                              activation='relu', 
                              name="{}_LSTM1".format(name)),
            layers.BatchNormalization(name="{}_BN1".format(name)),
            layers.ConvLSTM2D(filters=512,
                              kernel_size=(3, 3),
                              padding="same",
                              strides=(1, 1), 
                              return_sequences=True,
                              activation='relu', 
                              name="{}_LSTM2".format(name)),
            layers.BatchNormalization(name="{}_BN2".format(name)),
        ])

    def call(self, x):
        _, _, h, w, _ = x.shape
        x = self.model(x)
        if self.pooling:
            x = tf.reshape(x, [-1, h, w, 512])
            x = layers.GlobalAveragePooling2D()(x)
            x = tf.reshape(x, [-1, self.size, 1, 1, 512])
        return x


class Classifier(layers.Layer):
    
    def __init__(self, name):
        super(Classifier, self).__init__()
        self.model = Sequential([
            layers.Flatten(),
            layers.Dense(units=512, name="{}_Dense1".format(name), activation='relu'), 
            layers.Dropout(rate=0.5),
            layers.Dense(units=1, name="{}_Dense2".format(name), activation='sigmoid'),
        ], name=name)
    
    def call(self, x):
        x = self.model(x)
        return x

if __name__=="__main__":

    model = LSTMBlock("Block1", 450, 1800, 5, True)
    model(Input(shape=(5, 112, 450, 3)))
    model.model.summary()
    exit()
    model = CNNBlock("Block1", 450, 1800, 5, True)
    model(Input(shape=(5, 450, 1800, 3)))
    model.resnet.summary()

    model = CNNBlock("Block1", 450, 1800, 5, False)
    model(Input(shape=(5, 450, 1800, 3)))
    model.resnet.summary()
