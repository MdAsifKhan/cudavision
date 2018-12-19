#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'November 2018'

import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


from model import CNN
from evaluator import ModelEvaluator

cudnn.benchmark = True


trainset = dsets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
testset = dsets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Parameters
n_out = len(classes)
batch_size = 128

# Hyperparameters
lr = 0.001
epochs = 15
# Data Loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)


# Model
l2 = 0.0
pool = 'max'
optim = 'adam'
# Pytorch Cross Entropy Loss
model = CNN(pool)
modeleval = ModelEvaluator(model, epochs, lr, l2=l2, use_gpu=True, optim=optim)
modeleval.evaluator(trainloader, testloader, print_every=100, validation=False)