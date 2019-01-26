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
from seq_dataset import BallDataset
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

dataset = BallDataset(opt.seq_dataset)

model, loss, optimizer = joined_model.create_model()
model.resume_sweaty(os.path.abspath(opt.sweaty_resume_str))
if opt.seq_resume:
    model.resume_seq(os.path.abspath(opt.seq_resume_str))

trainloader = torch.utils.data.DataLoader(dataset,
                                          batch_size=opt.batch_size,
                                          shuffle=True,
                                          num_workers=opt.workers)

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
                                         batch_size=opt.batch_size,
                                         shuffle=False,
                                         num_workers=opt.workers,
                                         drop_last=True)

# model = training(dataloader, opt.nm_epochs, save=opt.seq_save_model,
#                  model=model,
#                  optimizer=optimizer,
#                  loss=loss,
#                  name=opt.seq_model)


opt.lr = 1e-3
modeleval = ModelEvaluator(model, 0.5)
modeleval.evaluator(trainloader, testloader)
modeleval.plot_loss()

