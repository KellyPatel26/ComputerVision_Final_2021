from tensorflow.keras.utils import Sequence
from PIL import Image
import numpy as np   
import glob
import os


class KaggleTrainGenerator(Sequence):

    def __init__(self, path, batch, shuffle=False, total_frames=10, sample_frames=[0, 2, 4, 6, 8], h=224, w=224):
        self.batch = batch
        self.shuffle = shuffle
        self.sample_frames = sample_frames
        self.h = h
        self.w = w
        self.paths, self.idxs = self.load_ids(path, total_frames)

    def __len__(self):
        # number of iteration for one epoch
        # drop the last batch if non-divisible 
        self.on_epoch_begin()
        return np.floor(len(self.paths)/self.batch).astype(int)

    def __getitem__(self, idx):
        # get indexs for this batch
        idxs = self.idxs[idx*self.batch:(idx+1)*self.batch]
        self.xs, self.ys = self.__data_generation(idxs)
        return self.xs, self.ys

    def load_ids(self, path, total_frames):
        # get paths of files
        paths = glob.glob(os.path.join(path, '*', '*', '*.jpg'))
        paths = np.array(sorted(paths)).reshape(-1, total_frames)
        idxs = np.arange(len(paths))
        return paths, idxs

    def on_epoch_begin(self):
        if self.shuffle == True:
            np.random.shuffle(self.idxs)
    
    def __data_generation(self, idxs):
        # load dataset
        xs = np.zeros((self.batch, len(self.sample_frames), self.h, self.w, 3))
        #ys = np.zeros((self.batch, 1))
        ys = np.zeros((self.batch))
        for i, video in enumerate(self.paths[idxs]):
            for j, frame in enumerate(video[self.sample_frames]):
                xs[i, j, :] = np.array(Image.open(frame), dtype=np.float32)/255.0
            #ys[i, 0] = 1 if "FAKE" in video[self.sample_frames][0] else 0
            ys[i] = 1 if "FAKE" in video[self.sample_frames][0] else 0
        return xs, ys


class KaggleTrainBalanceGenerator(Sequence):

    def __init__(self, path, batch, shuffle=False, total_frames=10, sample_frames=[0, 2, 4, 6, 8], h=224, w=224):
        # we will resample the data from real and fake
        assert batch%2==0
        self.batch = batch
        self.shuffle = shuffle
        self.sample_frames = sample_frames
        self.h = h
        self.w = w
        self.real_paths, self.real_idxs = self.load_ids(os.path.join(path, 'REAL'), total_frames)
        self.fake_paths, self.fake_idxs = self.load_ids(os.path.join(path, 'FAKE'), total_frames)

    def __len__(self):
        # number of iteration for one epoch
        # drop the last batch if non-divisible
        self.on_epoch_begin()
        return np.floor(min(len(self.real_paths), len(self.fake_paths))/self.batch).astype(int)

    def __getitem__(self, idx):
        # get indexs for this batch
        real_idxs = self.real_idxs[idx*self.batch//2:(idx+1)*self.batch//2]
        fake_idxs = self.fake_idxs[idx*self.batch//2:(idx+1)*self.batch//2]
        self.xs, self.ys = self.__data_generation(real_idxs, fake_idxs)
        return self.xs, self.ys

    def load_ids(self, path, total_frames):
        # get paths of files
        paths = glob.glob(os.path.join(path, '*', '*.jpg'))
        paths = np.array(sorted(paths)).reshape(-1, total_frames)
        idxs = np.arange(len(paths))
        return paths, idxs

    def on_epoch_begin(self):
        if self.shuffle == True:
            np.random.shuffle(self.real_idxs)
            np.random.shuffle(self.fake_idxs)

    def __data_generation(self, real_idxs, fake_idxs):
        # load dataset
        xs = np.zeros((self.batch, len(self.sample_frames), self.h, self.w, 3))
        #ys = np.ones((self.batch, 1))
        ys = np.ones((self.batch))
        for i, video in enumerate(self.fake_paths[fake_idxs]):
            for j, frame in enumerate(video[self.sample_frames]):
                xs[i, j, :] = np.array(Image.open(frame), dtype=np.float32)/255.0
            #ys[i, 0] = 0

        for i, video in enumerate(self.real_paths[real_idxs]):
            for j, frame in enumerate(video[self.sample_frames]):
                xs[i+self.batch//2, j, :] = np.array(Image.open(frame), dtype=np.float32)/255.0
            #ys[i+self.batch//2, 0] = 0
            ys[i+self.batch//2] = 0
        return xs, ys


class KaggleTestGenerator(Sequence):

    def __init__(self, path, batch, shuffle=False, total_frames=10, sample_frames=[0, 2, 4, 6, 8], h=224, w=224):
        self.batch = batch
        self.shuffle = shuffle
        self.sample_frames = sample_frames
        self.h = h
        self.w = w
        self.paths, self.idxs = self.load_ids(path, total_frames)

    def __len__(self):
        # number of iteration for one epoch
        # drop the last batch if non-divisible 
        self.on_epoch_begin()
        return np.floor(len(self.paths)/self.batch).astype(int)

    def __getitem__(self, idx):
        # get indexs for this batch
        idxs = self.idxs[idx*self.batch:(idx+1)*self.batch]
        self.xs = self.__data_generation(idxs)
        return self.xs

    def load_ids(self, path, total_frames):
        # get paths of files
        paths = glob.glob(os.path.join(path, '*', '*.jpg'))
        paths = np.array(sorted(paths)).reshape(-1, total_frames)
        idxs = np.arange(len(paths))
        return paths, idxs

    def on_epoch_begin(self):
        if self.shuffle == True:
            np.random.shuffle(self.idxs)
    
    def __data_generation(self, idxs):
        # load dataset
        xs = np.zeros((self.batch, len(self.sample_frames), self.h, self.w, 3))
        for i, video in enumerate(self.paths[idxs]):
            for j, frame in enumerate(video[self.sample_frames]):
                xs[i, j, :] = np.array(Image.open(frame), dtype=np.float32)/255.0
        return xs


if __name__=="__main__":
    data_generator = KaggleTrainBalanceGenerator(path="./dataset/train/", batch=30, shuffle=True)
    for xs, ys in data_generator:
        #print(xs)
        print(ys)

    data_generator = KaggleTrainGenerator(path="./dataset/val/", batch=30, shuffle=True)
    for xs, ys in data_generator:
        #print(xs)
        print(ys)

    data_generator = KaggleTestGenerator(path="./dataset/test/", batch=30, shuffle=True)
    for xs in data_generator:
        print(xs)
