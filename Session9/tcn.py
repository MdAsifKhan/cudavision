#!/usr/bin/env python

"""Implementation of simple tcn model
"""

__author__ = 'Anna Kukleva'
__date__ = 'December 2018'

import torch
import torch.nn as nn
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os

from arguments import opt
from logging_setup import logger
from tcn_locuslab import TemporalConvNet
from util_functions import Averaging


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self._init_weights()

    def _init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        out = self.tcn(x)
        return self.linear(out[:, :, -1])


def create_model():
    torch.manual_seed(opt.seed)
    # channel_sizes = [opt.nhid] * opt.levels
    channel_sizes = []
    for i in range(opt.levels):
        channel_sizes.append(opt.nhid * 2 ** i)
    model = TCN(input_size=opt.map_size * opt.map_size,
                output_size=opt.output_size,
                num_channels=channel_sizes,
                kernel_size=opt.ksize,
                dropout=opt.dropout).cuda()
    loss = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer


def test(dataloader, model):
    loss = Averaging()
    window = signal.gaussian(opt.window_size, std=4).reshape((-1, 1))
    window = np.dot(window, window.T)
    idx = 0
    with torch.no_grad():
        for i, (features, coord) in enumerate(dataloader):
            # if i % 10:
            #     continue
            gt_x, gt_y = coord.numpy().squeeze()
            gt_x, gt_y = int(gt_x), int(gt_y)
            x_size = opt.window_size
            y_size = opt.window_size
            heatmap_gt = np.zeros((opt.map_size_x, opt.map_size_y), dtype=np.float32)
            if gt_x + opt.window_size  > opt.map_size_x:
                x_size = max(0, opt.map_size_x - gt_x)
            if gt_y + opt.window_size  > opt.map_size_y:
                y_size = max(0, opt.map_size_y - gt_y)
            heatmap_gt[gt_x:gt_x + x_size, gt_y:gt_y + y_size] = window[:x_size, :y_size]

            if opt.real_balls:
                features = features.squeeze()
            features = features.float().cuda()
            pr_x, pr_y = model(features).cpu().numpy().squeeze()
            sweaty_out, out23 = model.test(features, ret_out23=True)
            sweaty_out = sweaty_out.cpu().numpy()[-1]
            out23 = out23.cpu().numpy()[-1]
            pr_x, pr_y = int(pr_x), int(pr_y)
            heatmap_pr = np.zeros((opt.map_size_x, opt.map_size_y), dtype=np.float32)
            x_size = opt.window_size
            y_size = opt.window_size
            if pr_x + opt.window_size  > opt.map_size_x:
                x_size = opt.map_size_x - pr_x
            if pr_y + opt.window_size  > opt.map_size_y:
                y_size = opt.map_size_y - pr_y
            heatmap_pr[pr_x:pr_x + x_size, pr_y:pr_y + y_size] = window[:x_size, :y_size]

            pr = [pr_x, pr_y]
            gt = [gt_x, gt_y]
            mse = int(np.sqrt(np.sum(np.square(np.array(pr) - np.array(gt)))))
            loss.update(mse)
            logger.debug('pr / gt:  %s / %s  |  %s' % (str(pr), str(gt), str(mse)))
            # logger.debug('target : %s' % str([gt_x, gt_y]))



            line = np.ones((heatmap_pr.shape[0], 5)) * np.max(heatmap_gt)
            concat = np.hstack((heatmap_pr, line, heatmap_gt, line, sweaty_out, line,out23))
            plt.axis('off')
            pltimg = plt.imshow(concat)
            plt.savefig(os.path.join(opt.save_out, opt.seq_model, 'out%d.pr_gt.png' % idx))

            idx += 1
            # if idx == 100:
            #     break
