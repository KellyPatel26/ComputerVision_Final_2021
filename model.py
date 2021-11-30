"""
Computer Vision Final Project
Deepfake Classification Model
CS1430 - Computer Vision
Brown University
"""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model, optimizers, losses, Sequential, Input
from module import LSTMBlock, Classifier, CNNBlock


class CNNDeepFakeModel(Model):

    def __init__(self, args, h, w, size):
        super(CNNDeepFakeModel, self).__init__()
    
        # optimizer
        self.optimizer = optimizers.Adam(learning_rate=args.lr)

        # architexture
        self.architecture = Sequential([
            CNNBlock(name="CNN1", h=h, w=w, size=size, pooling=True),
            Classifier(name="classifier")
        ])
    
    def call(self, x):
        # x: (batch, frames, height, width, color-channel)
        x = self.architecture(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        # labels: (batch)
        # predictions: (batch, 1)
        return losses.BinaryCrossentropy(from_logits=False)(labels, predictions)


class LSTMDeepFakeModel(Model):

    def __init__(self, args, h, w, size):
        super(LSTMDeepFakeModel, self).__init__()

        # optimizeir
        self.optimizer = optimizers.Adam(learning_rate=args.lr)

        # architexture
        self.architecture = Sequential([
            CNNBlock(name="CNN1", h=h, w=w, size=size, pooling=False),
            LSTMBlock(name="LSTM1", size=size, pooling=True),
            Classifier(name="classifier")
        ])

    def call(self, x):
        # x: (batch, frames, height, width, color-channel)
        x = self.architecture(x)
        return x
    
    def freezeCNN(self):
        for block in self.architecture.layers:
            if isinstance(block, CNNBlock):
                for layer in block.resnet.layers:
                    layer.trainable = False

    @staticmethod
    def loss_fn(labels, predictions):
        # labels: (batch)
        # predictions: (batch, 1)
        return losses.BinaryCrossentropy(from_logits=False)(labels, predictions)
