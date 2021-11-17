import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Sequential, Input, Model


class LSTMBlock(layers.Layer):
    
    def __init__(self, name):
        super(LSTMBlock, self).__init__()
        self.model = Sequential([
            layers.ConvLSTM2D(filters=64,
                              kernel_size=(3, 3),
                              strides=(16, 16),
                              padding="same",
                              return_sequences=False,
                              activation=None),
            ], name="{}_LSTMBlock".format(name))

    def call(self, x):
        x = self.model(x)
        return x


class Classifier(layers.Layer):
    
    def __init__(self, name):
        super(Classifier, self).__init__()
        self.model = Sequential([
            layers.Flatten(),
            layers.Dense(units=256, name="{}_Dense1".format(name), activation='relu'), 
            layers.Dropout(rate=0.0),
            layers.Dense(units=2, name="{}_Dense2".format(name), activation='softmax'),
        ], name="{}_Classifier".format(name))
    
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
