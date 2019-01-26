#!/usr/bin/env python

__author__ = 'Anna Kukleva'
__date__ = 'December 2018'


from torch.utils.data import Dataset
import torch
from torchvision import datasets, transforms
import numpy as np
from os.path import join
import os
import re
import h5py
import tqdm
from collections import defaultdict

from util_functions import join_data
from logging_setup import logger
from arguments import opt


def keyf(filename):
    if filename.endswith('txt'):
        return (-1, -1)
    search = re.search(r'ball(\d*)_(\d*)', filename)
    n_ball = int(search.group(1))
    n_frame = int(search.group(2))
    return (n_ball, n_frame)


class BallDataset(Dataset):
    def __init__(self, path, maxlen=20, prediction=10):
        logger.debug('create ball dataset')
        self.balls = {}
        self.balls_coord = {}
        self.balls_frames = []
        self.len = 0
        self.balls_maxframe = {}
        self.maxlen = maxlen
        if opt.seq_model == 'lstm':
            self.prediction = prediction
        if opt.seq_model == 'tcn':
            self.prediction = 1
        files = sorted(list(os.listdir(path)), key=keyf)
        for filename in files:
            if not filename.endswith('.npy'):
                continue
            search = re.search(r'ball(\d*)_(\d*).npy', filename)
            n_ball = int(search.group(1))
            n_frame = int(search.group(2))
            feature_map = np.load(join(path, filename))[..., None]
            features = self.balls.get(n_ball, None)
            features = join_data(features, feature_map, np.concatenate, axis=2)
            self.balls[n_ball] = features
        self.h = features.shape[0]
        self.w = features.shape[1]
        for ball_idx, data in self.balls.items():
            for n_frame in range(1, data.shape[-1] - self.prediction):
                self.balls_frames.append([ball_idx, n_frame])
            self.balls_coord[ball_idx] = np.loadtxt(os.path.join(path, 'ball%d.txt' % ball_idx))

    def __len__(self):
        return len(self.balls_frames)

    def __getitem__(self, idx):
        ball_idx, frame = self.balls_frames[idx]
        if frame > self.maxlen:
            seq = self.balls[ball_idx][..., frame - self.maxlen: frame]
        else:
            seq = np.zeros((self.h, self.w, self.maxlen))
            seq[..., -frame:] = self.balls[ball_idx][..., :frame]

        # seq_len = min(features.shape[-1], self.maxlen)
        # seq[..., :seq_len] = features[..., :seq_len]
        seq = seq.transpose(2, 0, 1)
        if opt.seq_model == 'lstm':
            next_steps = self.balls[ball_idx][..., frame: frame + self.prediction].transpose(2, 0, 1)
            return np.asarray(seq, dtype=float), np.asarray(next_steps, dtype=float)
        if opt.seq_model == 'tcn':
            seq = seq.reshape((-1, opt.hist))
            coords = self.balls_coord[ball_idx][frame]
            return np.asarray(seq, dtype=float), np.asarray(coords, dtype=float)


class RealBallDataset(Dataset):
    def __init__(self, data_path, map_file, transform=None):
        self.dataroot = data_path
        self.map_file = map_file
        self.transform = transform

        with h5py.File(self.dataroot + '/' + self.map_file,'r') as hf:
            targets = hf['prob_maps'].value
            targets = np.array(targets).astype('float32')
            self.filenames = list(hf['filenames'].value)
            box = list(hf['ros'].value)

        self.threshold = 0.7 * targets.max()
        self.images, self.targets, self.box = [], [], []
        self.filenames = [filename.decode('utf-8') for filename in self.filenames]
        for filename in tqdm(os.listdir(self.dataroot)):
            name = filename[:-4]
            if filename.endswith('.jpg'):
                if name in self.filenames:
                    self.images.append(filename)
                    idx = self.filenames.index(name)
                    self.targets.append(targets[idx])
                    self.box.append(box[idx].astype('float32'))
                '''
                else:
                    self.targets.append(np.zeros([120, 160], dtype='float32'))
                    self.box.append(np.array([0, 0, 0, 0], dtype='float32'))
                '''
    def __len__(self):
        return len(self.targets)


    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.dataroot, img_name)
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        #img = np.asarray(img).transpose(2, 0, 1)/255.0
        #img = torch.from_numpy(img).float()
        prob_ = self.targets[idx]
        coord_ = self.box[idx]
        return img, prob_, coord_


if __name__ == '__main__':
    folder = 'toy.seq'
    dataset = BallDataset(folder)
    features, gt = dataset[38]
    logger.debug('done')


