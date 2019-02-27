#!/usr/bin/env python

""" Join sweatynet and sequential part
"""

__author__ = 'Anna Kukleva'
__date__ = 'January 2019'

import torch
import torch.nn as nn
import os
from torch.autograd import Variable

from arguments import opt
from model import SweatyNet1, SweatyNet2, SweatyNet3
from lstm import LSTM
from logging_setup import logger
from tcn import TCN
from tcn_ed import TCN_ED

def init_weights(m):
    if type(m) == nn.Conv1d:
        m.weight.data.normal_(0, 0.01)


class Weight(nn.Module):
    def __init__(self):
        super(Weight, self).__init__()
        self.multp = nn.Parameter(torch.Tensor([0.5]))


class JoinedModel(nn.Module):
    def __init__(self):
        super(JoinedModel, self).__init__()
        self.sweaty = SweatyNet1(1, opt.drop_p, finetune=True)
        if opt.seq_model == 'lstm':
            self.seq = LSTM()
        if opt.seq_model == 'tcn':
            n_nodes = [64, 96]
            self.seq = TCN_ED(n_nodes, opt.hist, opt.seq_predict, opt.ksize).to(opt.device)
            self.seq.apply(init_weights)
            # channel_sizes = []
            # for i in range(opt.levels):
            #     channel_sizes.append(opt.nhid * 2 ** i)
            # self.seq = TCN(input_size=opt.map_size_x * opt.map_size_y,
            #                output_size=opt.output_size,
            #                num_channels=channel_sizes,
            #                kernel_size=opt.ksize,
            #                dropout=opt.dropout)
            #
        self.conv1 = nn.Sequential(nn.Conv2d(112, 1, 7, padding=3),
                                   nn.BatchNorm2d(1),
                                   nn.LeakyReLU())
        # self.conv1 = nn.Sequential(nn.Conv2d(112, 1, 1, padding=0),
        #                            nn.BatchNorm2d(1),
        #                            nn.LeakyReLU())
        # self.conv1 = nn.Sequential(nn.Conv2d(25, 28, 7, padding=3),
        #                            # nn.BatchNorm2d(28),
        #                            nn.LeakyReLU())
        # self.conv2 = nn.Sequential(nn.Conv2d(28, 14, 5, padding=2),
        #                            # nn.BatchNorm2d(14),
        #                            nn.LeakyReLU())
        # self.conv3 = nn.Sequential(nn.Conv2d(14, 1, 3, padding=1),
        #                            # nn.BatchNorm2d(1),
        #                            nn.LeakyReLU())
        # self.alpha = torch.nn.Parameter(torch.Tensor([0.05]))
        # self.alpha = torch.Tensor([0.05])
        # self.alpha = Variable(torch.Tensor([0.05]).cuda(), requires_grad=True)
        # self.alpha = self.alpha.cuda()
        self.alpha = Weight()



    def forward(self, x):
        x, out23 = self.sweaty(x)
        out23 = self.alpha.multp * self.conv1(out23).squeeze()
        # out23 = 0.05 * self.conv1(out23).squeeze()
        x = x + out23
        if opt.seq_model == 'tcn':
            # out23 = torch.cat((x.unsqueeze(1), out23), 1)
            # out23 = self.conv3(self.conv2(self.conv1(out23))).squeeze()
            # x = self.alpha * x + (1 - self.alpha) * out23

            # out23 = self.alpha.multp * self.conv1(out23).squeeze()
            # out23 = 0.05 * self.conv1(out23).squeeze()
            # x = x + out23
            # x = x + (1 - self.alpha) * out23
            x = x.view(-1, opt.hist, opt.map_size_x * opt.map_size_y)
            # x = (x - torch.mean(x, dim=1, keepdim=True)) / (torch.var(x, dim=1, keepdim=True))
        if opt.seq_model == 'lstm':
            x = x.view(-1, opt.hist, opt.map_size_x, opt.map_size_y)
        x = self.seq(x)
        return x

    def test(self, x, ret_out23=False):
        x, out23 = self.sweaty(x)
        if ret_out23:
            out23 = self.conv1(out23).squeeze()
            return x, out23
        return x

    def resume_sweaty(self, path):
        checkpoint = torch.load(path)
        checkpoint = checkpoint['state_dict_model']
        self.sweaty.load_state_dict(checkpoint)

    def resume_seq(self, path):
        checkpoint = torch.load(path)
        checkpoint = checkpoint['state_dict']
        self.seq.load_state_dict(checkpoint)

    def resume_both(self, path):
        checkpoint = torch.load(path)
        checkpoint = checkpoint['state_dict_model']
        self.load_state_dict(checkpoint)

    def off_sweaty(self):
        for param in self.sweaty.parameters():
            param.requires_grad = False

    def on_sweaty(self):
        for param in self.sweaty.parameters():
            param.requires_grad = True


def create_model():
    torch.manual_seed(opt.manualSeed)
    model = JoinedModel()
    if not opt.seq_resume:
        model.off_sweaty()
    if opt.use_gpu:
        model = model.cuda()
    loss = nn.MSELoss(reduction='sum')
    # loss = nn.BCELoss()
    optimizer_seq = torch.optim.Adam(list(model.seq.parameters()) +
                                 list(model.conv1.parameters()) +
                                 list(model.alpha.parameters()),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

    params = list(model.seq.parameters()) + list(model.conv1.parameters()) + list(model.alpha.parameters())
    both_lr = opt.lr if opt.seq_resume else opt.lr * 0.1
    optimizer_both = torch.optim.Adam([
        {'params': model.sweaty.parameters(), 'lr': both_lr},
        {'params': params}
                                       ],
                                      lr=opt.lr,
                                      weight_decay=opt.weight_decay)
    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer_seq))
    logger.debug(str(optimizer_both))

    return model, loss, optimizer_seq, optimizer_both





