import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Sequential, Input, Model


class LSTMBlock(layers.Layer):
    
    def __init__(self, name):
        super(LSTMBlock, self).__init__()
        self.model = Sequential([
            layers.ConvLSTM2D(filters=64,
                              kernel_size=(5, 5),
                              padding="same",
                              strides=(1, 1), 
                              return_sequences=True,
                              activation='relu', 
                              name="{}_LSTM1".format(name)),
            layers.BatchNormalization(name="{}_BN1".format(name)),
            layers.ConvLSTM2D(filters=64,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              padding="same",
                              return_sequences=True,
                              activation="relu",
                              name="{}_LSTM2".format(name)),
            layers.BatchNormalization(name="{}_BN2".format(name)),
            layers.ConvLSTM2D(filters=64,
                              kernel_size=(1, 1),
                              strides=(2, 2),
                              padding="same",
                              return_sequences=True,
                              activation="relu",
                              name="{}_LSTM3".format(name)),
            layers.Conv3D(filters=1, 
                          kernel_size=(3, 3, 3), 
                          activation="relu", 
                          padding="same",
                          name="{}_Conv3D1".format(name))
            ], name="{}_LSTMBlock".format(name))

    def call(self, x):
        x = self.model(x)
        return x


class Classifier(layers.Layer):
    
    def __init__(self, name):
        super(Classifier, self).__init__()
        self.model = Sequential([
            layers.Flatten(),
            layers.Dense(units=512, name="{}_Dense1".format(name), activation='relu'), 
            layers.Dropout(rate=0.0),
            layers.Dense(units=1, name="{}_Dense2".format(name), activation='sigmoid'),
        ], name=name)
    
    def call(self, x):
        x = self.model(x)
        return x

if __name__=="__main__":
    model1 = LSTMBlock("Block1")
    model1(Input(shape=(5, 224, 224, 3)))
    model1.model.summary()

    model2 = Classifier("Block1")
    model2(Input(shape=(5, 224, 224, 3)))
    model2.model.summary()
