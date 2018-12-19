#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'December 2018'

from torch.utils.data import Dataset
import torch
import os
import re
import numpy as np
# from skimage import io, transform
# from torchvision import transforms
from PIL import Image

from utils.arg_pars import opt
from utils.logging_setup import logger


def valid_frame_parser():
    valid_frames = {}
    total = 0
    for filename in os.listdir(opt.valid_frames):
        frames = []
        if 'NimbRo-OP2' in filename:
            fps = 24
        else:
            fps = 30
        with open(os.path.join(opt.valid_frames, filename)) as f:
            for line in f:
                search = re.search(r'(\d*).(\d*)-(\d*).(\d*)', line)
                start_min = int(search.group(1))
                start_sec = int(search.group(2))
                end_min = int(search.group(3))
                end_sec = int(search.group(4))
                start = (start_min * 60 + start_sec) * fps
                end = (end_min * 60 + end_sec) * fps
                frames += list(range(start, end))
        valid_frames[filename] = frames
        total += len(frames)
    print('total: %d' % total)
    return valid_frames


class SoccerDataset(Dataset):
    def __init__(self, transform=None):
        np.random.seed(opt.seed)
        self.transform = transform

        self.soccer_folders = opt.save_folder
        self.pascal = opt.pascal

        self.valid_frames = valid_frame_parser()
        self.image_pathes = []
        self.labels = []
        for video_name in self.valid_frames:
            sample_fraction = len(self.valid_frames[video_name]) * opt.sample_rate
            sampled_frames = np.random.choice(self.valid_frames[video_name],
                                              int(sample_fraction),
                                              replace=False)
            cur_path = os.path.join(self.soccer_folders, video_name)
            for i in sampled_frames:
                self.image_pathes.append(os.path.join(cur_path, '%d.png' % i))
                self.labels.append(1)

        logger.debug('%d soccers' % len(self.image_pathes))

        for file_idx, filename in enumerate(os.listdir(self.pascal)):
            self.image_pathes.append(os.path.join(self.pascal, filename))
            self.labels.append(0)
            if file_idx > 7000:
                break

        logger.debug('%d with pascal' % len(self.image_pathes))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # img = io.imread(self.image_pathes[idx])
        # img = transforms.ToTensor(img)
        img = Image.open(self.image_pathes[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]







