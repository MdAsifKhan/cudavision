#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'December 2018'

import torch.nn as nn
import torch
from torchvision import models, transforms

from soccer_dataset import SoccerDataset
from training_network import training
from utils.arg_pars import opt
from utils.logging_setup import logger, path_logger
from utils.utils import update_opt_str


def logger_setup():
    update_opt_str()
    path_logger()


def train():
    logger.debug('.')
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.cuda()
    loss = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    dataset = SoccerDataset(transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.num_workers)

    training(dataloader, model=model, loss=loss, optimizer=optimizer)

def test():
    pass


if __name__ == '__main__':
    logger_setup()
    train()
