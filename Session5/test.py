#!/usr/bin/env python

"""
"""


import torch.nn as nn
import torch
from torchvision import models, transforms
import os
import numpy as np

from soccer_dataset import SoccerDataset
from training_network import training, load_model
from utils.arg_pars import opt
from utils.logging_setup import logger, path_logger
from utils.utils import update_opt_str
from PIL import Image


def logger_setup():
    update_opt_str()
    path_logger()


def test_img(path):
    logger_setup()

    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = transform(img)
    img = img.numpy()
    img = img[np.newaxis, ...]
    img = torch.Tensor(img)

    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model.load_state_dict(load_model(epoch=9))
    model.eval()

    with torch.no_grad():
        output = model(img).numpy()
        output = np.exp(output) / np.sum(np.exp(output))
        logger.debug(str(output))
        idx = np.argmax(output)
        logger.debug('prediction: %s' % ['not soccer', 'soccer'][idx])



if __name__=='__main__':
    path = os.path.join(opt.test, 'test4.jpg')
    test_img(path)
