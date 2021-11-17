"""
CSCI 1430 Computer Vision Final project
By Cheng-You Lu, Ji Won Chung, Kelly Patel

Usage:
    
"""
import os
import argparse
import datetime
from tensorflow.keras import callbacks, Input
from preprocess import KaggleTrainGenerator, KaggleTrainBalanceGenerator, KaggleTestGenerator
from model import LSTMDeepFakeModel

parser = argparse.ArgumentParser(description='DeepFakeClassifier')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='./dataset/', help='path of the dataset')
parser.add_argument('--type', dest='type', default='LSTM', help='LSTM')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='# of video for a batch')
parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='# epoch')
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
                                  period=10)
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
    print("aaaaaa")
    # test on dataset with labels
    model.evaluate_generator(
        generator=test_generator,
        verbose=1,
    )


def main():
    # change later
    init_epoch = 0
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    
    # reuse the directory if loading checkpoint
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        timestamp = args.load_checkpoint.split(os.sep)[-2]
    
    # some hyper-parameters related to dataset are here
    sample = [0, 2, 4, 6, 8]
    total_frames = 10
    h, w = 224, 224
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
    save_dir = "./checkpoints/" 
    log_dir = "./logs/"
    # specify model
    if args.type == 'LSTM':
        model = LSTMDeepFakeModel(args)
        model(Input(shape=(len(sample), 224, 224, 3)))
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
        model.load_weights(args.load_checkpoint)
    
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
