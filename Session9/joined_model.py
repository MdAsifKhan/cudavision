#!/usr/bin/env python

""" Join sweatynet and sequential part
"""

__author__ = 'Anna Kukleva'
__date__ = 'January 2019'

import torch
import torch.nn as nn


from arguments import opt
from model import SweatyNet1
from lstm import LSTM
from logging_setup import logger
from tcn import TCN

class JoinedModel(nn.Module):
    def __init__(self):
        super(JoinedModel, self).__init__()
        self.sweaty = SweatyNet1(1, opt.drop_p)
        if opt.seq_model == 'lstm':
            self.seq = LSTM()
        if opt.seq_model == 'tcn':
            channel_sizes = [opt.nhid] * opt.levels
            self.seq = TCN(input_size=opt.map_size_x * opt.map_size_y,
                           output_size=opt.output_size,
                           num_channels=channel_sizes,
                           kernel_size=opt.ksize,
                           dropout=opt.dropout)

    def forward(self, x):
        x = self.sweaty(x)
        x = x.view(1, -1, opt.hist)
        x = self.seq(x)
        return x

    def test(self, x):
        x = self.sweaty(x)
        return x

    def resume_sweaty(self, path):
        checkpoint = torch.load(path)
        checkpoint = checkpoint['state_dict_model']
        self.sweaty.load_state_dict(checkpoint)

    def resume_seq(self, path):
        checkpoint = torch.load(path)
        checkpoint = checkpoint['state_dict']
        self.seq.load_state_dict(checkpoint)


def create_model():
    torch.manual_seed(opt.manualSeed)
    model = JoinedModel()
    if opt.use_gpu:
        model = model.cuda()
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))

    return model, loss, optimizer





