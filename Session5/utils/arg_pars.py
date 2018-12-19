#!/usr/bin/env python

"""Hyper-parameters and logging set up

opt: include all hyper-parameters
logger: unified logger for the project
"""

__all__ = ['opt']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import argparse
from os.path import join

parser = argparse.ArgumentParser()

###########################################
parser.add_argument('--dataset_root', default='/media/data/kukleva/Study/CudaVisionLab/assignment5',
                    help='root folder for dataset')
parser.add_argument('--save_folder', default='/media/data/kukleva/Study/CudaVisionLab/assignment5/frames')
parser.add_argument('--video_folder', default='/media/data/kukleva/Study/CudaVisionLab/assignment5/videos/')
parser.add_argument('--valid_frames', default='/media/data/kukleva/Study/CudaVisionLab/assignment5/valid_frames')
parser.add_argument('--pascal', default='/media/data/kukleva/datasets/PASCAL/VOCtrainval/VOC2012/JPEGImages')
parser.add_argument('--test', default='/media/data/kukleva/Study/CudaVisionLab/assignment5/test')

###########################################
# hyperparams parameters for embeddings

parser.add_argument('--seed', default=42,
                    help='seed for random algorithms, everywhere')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_adj', default=True, type=bool,
                    help='will lr be multiplied by 0.1 in the middle')
parser.add_argument('--weight_decay', default=1e-4,
                    help='regularization constant for l_2 regularizer of W')
parser.add_argument('--batch_size', default=64,
                    help='batch size for training embedding (default: 40)')
parser.add_argument('--num_workers', default=7,
                    help='number of threads for dataloading')
parser.add_argument('--epochs', default=15, type=int,
                    help='number of epochs for training embedding')

###########################################
# save
parser.add_argument('--save_model', default=True, type=bool,
                    help='save embedding model after training')
parser.add_argument('--resume', default=False, type=bool,
                    help='load model for embeddings, if positive then it is number of '
                         'epoch which should be loaded')
parser.add_argument('--resume_str',
                    default='',
                    # default='/media/data/kukleva/Weak_YouTube/models/3500_relu_bias0.01_dim1024_ep15_im0_iv0.001_lr0.001_15',
                    )

###########################################
# additional
parser.add_argument('--sample_rate', default=0.5, type=int,
                    help='process full (if value is 0) /only part of the dataset')
parser.add_argument('--prefix', default='res18.pascal.reduced.',
                    help='prefix for log file')

###########################################
# logs
parser.add_argument('--log', default='DEBUG',
                    help='DEBUG | INFO | WARNING | ERROR | CRITICAL ')
parser.add_argument('--log_str', default='',
                    help='unify all savings')

opt = parser.parse_args()

