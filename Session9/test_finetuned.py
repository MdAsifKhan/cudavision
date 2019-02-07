#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'January 2019'


import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import random

from arguments import opt
from logging_setup import path_logger
from seq_dataset import BallDataset, RealBallDataset
import lstm
from training_net import training
from util_functions import dir_check
import joined_model
from evaluator import ModelEvaluator
from dataset import SoccerDataSet


path_logger()
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

testset = SoccerDataSet(data_path=opt.data_root + '/test_cnn', map_file='test_maps',
                        transform=transforms.Compose([
                            # transforms.RandomResizedCrop(opt.input_size[1]),
                            # transforms.RandomHorizontalFlip(),
                            # transforms.RandomRotation(opt.rot_degree),
                            transforms.ColorJitter(brightness=0.3,
                                                   contrast=0.4, saturation=0.4),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ]))

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=5,
                                         shuffle=False,
                                         num_workers=opt.workers,
                                         drop_last=True)

opt.batch_size = opt.hist

model, loss, optimizer = joined_model.create_model()
if opt.seq_both_resume:
    model.resume_both(os.path.abspath(opt.seq_both_resume_str))
else:
    model.resume_sweaty(os.path.abspath(opt.sweaty_resume_str))
    if opt.seq_resume:
        model.resume_seq(os.path.abspath(opt.seq_resume_str))

opt.lr = 1e-5
modeleval = ModelEvaluator(model, threshold=5.0535, min_radius=2.625)
modeleval.test(0, testloader)

