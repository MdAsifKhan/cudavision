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
from scipy import signal
import h5py
import tqdm
from collections import defaultdict
from PIL import Image

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
    def __init__(self, data_path, transform=None, prediction=20, small=False):
        self.dataroot = data_path
        # self.map_file = map_file
        self.transform = transform
        # self.ball_frame2idx = {}
        self._small = small

        if opt.seq_model == 'lstm':
            self.prediction = prediction
        if opt.seq_model == 'tcn':
            self.prediction = 1

        # with h5py.File(self.dataroot + '/' + self.map_file,'r') as hf:
        #     targets = hf['prob_maps'].value
        #     targets = np.array(targets).astype('float32')
        #     filenames = list(hf['filenames'].value)
        #     box = list(hf['ros'].value)

        self.threshold = 0.7
        # self.images, self.targets, self.box = [], [], []
        # filenames = [filename.decode('utf-8') for filename in filenames]

        self._read_seq()
        # for ball_idx, i in self.ball_frames:
        #     ball_filename, _ = self.balls[ball_idx][i]
            # self.images.append(ball_filename + '.jpg')
            # idx = filenames.index(ball_filename)
            # self.ball_frame2idx[i] = idx
            # self.targets.append(targets[idx])
            # self.box.append(box[idx].astype('float32'))

    def _read_seq(self):
        self.balls = {}
        self.ball_frames = []
        self.filenames = []
        if self._small:
            folder_name = 'balls_wo_zeros'
        else:
            folder_name = 'balls'
        for filename in os.listdir(os.path.join(opt.seq_real_balls, folder_name)):
            if 'ball' in filename:
                # balls_files.append(os.path.join(opt.seq_real_balls, filename))
                # ball_filename = os.path.join(opt.seq_real_balls, filename)
                ball_idx = len(self.balls)
                with open(os.path.join(opt.seq_real_balls, folder_name, filename), 'r') as f:
                    ball = []
                    for line in f:
                        line = list(map(lambda x: int(x), line.split()))
                        ball_filename = 'imageset_%d/frame%04d.jpg' % (line[0], line[1])
                        ball_center = line[-2:]
                        ball.append([ball_filename, ball_center])
                        self.filenames.append(ball_filename)
                    self.balls[ball_idx] = ball
                    for i in range(1, len(ball) - self.prediction):
                        self.ball_frames.append([ball_idx, i])

    def __len__(self):
        return len(self.ball_frames)

    def __getitem__(self, idx):
        ball_idx, iframe = self.ball_frames[idx]
        filenames = []
        # predict next coordinate of the ball
        frame = iframe
        _, gt_center = self.balls[ball_idx][frame]
        # collect sequence of the data back from the past
        while frame and len(filenames) != opt.hist:
            frame -= 1
            fn, _ = self.balls[ball_idx][frame]
            filenames.append(fn)

        seq = None
        for img_name in filenames:
            img_path = os.path.join(self.dataroot, img_name)
            img = Image.open(img_path)

            if self.transform:
                img = self.transform(img).view(1, 3, 480, 640)
            if seq is None:
                seq = img
            else:
                seq = torch.cat((img, seq))
        if len(filenames) < opt.hist:
            seq_z = torch.zeros(opt.hist - len(filenames), img.shape[1], img.shape[2], img.shape[3])
            seq = torch.cat((seq_z, seq))
        if opt.seq_model == 'tcn':
            return seq, np.asarray(gt_center, dtype=float)
        if opt.seq_model == 'lstm':
            heatmap = np.zeros((opt.map_size_x, opt.map_size_y), dtype=np.float32)
            window = signal.gaussian(opt.window_size, std=3).reshape((-1, 1))
            x,y = gt_center
            heatmap[x:x + opt.window_size, y:y + opt.window_size] = window
            return seq, heatmap
        #img = np.asarray(img).transpose(2, 0, 1)/255.0
        #img = torch.from_numpy(img).float()
        # prob_ = self.targets[idx]
        # coord_ = self.box[idx]


if __name__ == '__main__':
    folder = 'toy.seq'
    dataset = BallDataset(folder)
    features, gt = dataset[38]
    logger.debug('done')


