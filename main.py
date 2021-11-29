"""
CSCI 1430 Computer Vision Final project
By Cheng-You Lu, Ji Won Chung, Kelly Patel

Usage:
    
"""
import os
import argparse
import datetime
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras import callbacks, Input, optimizers
from preprocess import KaggleTrainGenerator, KaggleTrainBalanceGenerator, KaggleTestGenerator
from model import LSTMDeepFakeModel, CNNDeepFakeModel
#os.environ['CUDA_VISIBLE_DEVICES'] = ""

parser = argparse.ArgumentParser(description='DeepFakeClassifier')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='./dataset/', help='path of the dataset')
parser.add_argument('--type', dest='type', default='CNN', help='LSTM/CNN/LSTM-F')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='# of video for a batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=200, help='# epoch')
parser.add_argument('--load_pretrain', dest='load_pretrain', default=None, help='load pretrain')
parser.add_argument('--load_checkpoint', dest='load_checkpoint', default=None, help='load checkpoint')
args = parser.parse_args()


def train(model, train_generator, val_generator, checkpoint_path, logs_path, init_epoch=0):
    # keras callbacks tensorboard and saver for training
    callback_list = [
        callbacks.TensorBoard(log_dir=logs_path, 
                              update_freq='batch', 
                              profile_batch=0),
        callbacks.ModelCheckpoint(filepath=checkpoint_path+"/{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                  save_weights_only=True,
                                  monitor='val_accuracy',
                                  mode='max',
                                  save_best_only=True,
                                  period=1)
        ]

    # training
    model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        epochs=args.epoch,
        callbacks=callback_list,
        initial_epoch=init_epoch
    )


def test(model, test_generator):
    # test on dataset with labels
    acc = model.evaluate_generator(
            generator=test_generator,
            verbose=1,
        )
    print(acc)
    """
    pred = model.predict_generator(
        generator=test_generator, verbose=1,
    )
    """


def main():
    init_epoch = 0
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    # reuse the directory if loading checkpoint
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        timestamp = args.load_checkpoint.split(os.sep)[-2]
    
    save_dir = "./checkpoints/" 
    log_dir = "./logs/"
    # specify model
    if args.type == 'CNN':
        # some hyper-parameters related to dataset are here
        sample = [0, 2, 4, 6, 8]
        total_frames = 10
        h, w = 450, 1800
        train_generator = KaggleTrainBalanceGenerator(path=os.path.join(args.dataset_dir, "train"), 
                                                      batch=args.batch_size, 
                                                      shuffle=True,
                                                      sample_frames=sample,
                                                      total_frames=total_frames,
                                                      h=h,
                                                      w=w)
        val_generator = KaggleTrainGenerator(path=os.path.join(args.dataset_dir, "val"), 
                                             batch=args.batch_size, 
                                             shuffle=False,
                                             sample_frames=sample,
                                             total_frames=total_frames,
                                             h=h,
                                             w=w)
        model = CNNDeepFakeModel(args, h, w, len(sample))
        model(Input(shape=(len(sample), h, w, 3)))
        model.summary()
    elif args.type == 'LSTM':
        # some hyper-parameters related to dataset are here
        sample = [0, 2, 4, 6, 8]
        total_frames = 10
        h, w = 450, 1800 # 2500, 2500
        train_generator = KaggleTrainBalanceGenerator(path=os.path.join(args.dataset_dir, "train"), 
                                                      batch=args.batch_size, 
                                                      shuffle=True,
                                                      sample_frames=sample,
                                                      total_frames=total_frames,
                                                      h=h,
                                                      w=w)
        val_generator = KaggleTrainGenerator(path=os.path.join(args.dataset_dir, "val"), 
                                             batch=args.batch_size, 
                                             shuffle=False,
                                             sample_frames=sample,
                                             total_frames=total_frames,
                                             h=h,
                                             w=w)
        model = LSTMDeepFakeModel(args, h, w, len(sample))
        model(Input(shape=(len(sample), h, w, 3)))
        model.summary()
    elif args.type == 'LSTM-F':
        # some hyper-parameters related to dataset are here
        sample = [0, 2, 4, 6, 8]
        total_frames = 10
        h, w = 450, 1800
        train_generator = KaggleTrainBalanceGenerator(path=os.path.join(args.dataset_dir, "train"), 
                                                      batch=args.batch_size, 
                                                      shuffle=True,
                                                      sample_frames=sample,
                                                      total_frames=total_frames,
                                                      h=h,
                                                      w=w)
        val_generator = KaggleTrainGenerator(path=os.path.join(args.dataset_dir, "val"), 
                                             batch=args.batch_size, 
                                             shuffle=False,
                                             sample_frames=sample,
                                             total_frames=total_frames,
                                             h=h,
                                             w=w)
        model = LSTMDeepFakeModel(args, h, w, len(sample))
        model(Input(shape=(len(sample), h, w, 3)))
        opts = [
            optimizers.Adam(learning_rate=args.lr*0.1),
            optimizers.Adam(learning_rate=args.lr)
        ]
        optimizers_and_layers = [(opts[0], model.layers[0].layers[0]),
                                 (opts[1], model.layers[0].layers[1:])]
        model.optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        #model.freezeCNN()
        model.summary()


    checkpoint_path = os.path.join(save_dir, args.type, timestamp)
    logs_path = os.path.join(log_dir, args.type, timestamp)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)   
    
    # load checkpoints
    if args.load_checkpoint:
        assert os.path.split(args.load_checkpoint)[0]==checkpoint_path
        model.load_weights(args.load_checkpoint, by_name=True, skip_mismatch=True)
        init_epoch = int(os.path.split(args.load_checkpoint)[-1].split("-")[0])

    # compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["accuracy"])
    
    if args.phase=='train':
        train(model, train_generator, val_generator, checkpoint_path, logs_path, init_epoch)
    else:
        test(model, val_generator)


if __name__ == '__main__':
    main()
