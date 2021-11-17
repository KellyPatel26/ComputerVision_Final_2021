"""
Computer Vision Final Project
Deepfake Classification Model
CS1430 - Computer Vision
Brown University
"""
import tensorflow as tf
from tensorflow.keras import Model, optimizers, losses, Sequential
from module import LSTMBlock, Classifier
    
class LSTMDeepFakeModel(Model):

    def __init__(self, args):
        super(LSTMDeepFakeModel, self).__init__()
    
        # optimizer
        self.optimizer = optimizers.Adam(learning_rate=args.lr)

        # architexture
        self.architecture = Sequential([
            LSTMBlock(name="block1"),
            Classifier(name="classifier")
        ])
    
    def call(self, x):
        # x: (batch, frames, height, width, color-channel)
        x = self.architecture(x)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        # labels: (batch, 1)
        # predictions: (batch, 2)
        return losses.SparseCategoricalCrossentropy(from_logits=False)(labels, predictions)
