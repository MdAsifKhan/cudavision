import torch
import torch.nn as nn
from torch.nn import functional as F


import numpy as np
import matplotlib.pyplot as plt
import os

from convolution_lstm import ConvLSTM
from logging_setup import logger
from arguments import opt


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.clstm = ConvLSTM(input_channels=20,
                              hidden_channels=[64, 128, 64, 10],
                              kernel_size=5,
                              step=9,
                              effective_step=[8])

    def forward(self, x):
        x = self.clstm(x)
        return x


def create_model():
    torch.manual_seed(opt.seed)
    model = LSTM()
    loss = nn.MSELoss(reduction='sum').cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    logger.debug(str(model))
    logger.debug(str(loss))
    logger.debug(str(optimizer))
    return model, loss, optimizer


def test(dataloader, model):
    idx = 0
    with torch.no_grad():
        for i, (features, next_fr) in enumerate(dataloader):
            if i % 10:
                continue
            features = features.float().cuda()
            outputs, (h, cc) = model(features)
            outputs = outputs[0].cpu().numpy().squeeze()
            next_fr = next_fr.numpy().squeeze()
            for i_inner in range(outputs.shape[0]):
                concat = np.hstack((outputs[i_inner, ...], next_fr[i_inner, ...]))
                pltimg = plt.imshow(concat)
                plt.savefig(os.path.join(opt.save_out, opt.model, 'out%d.%d.pr_gt.png' % (idx, i_inner)))
                # pltimg = plt.imshow(cc[i_inner, ...])
                # plt.savefig(os.path.join(opt.save_out, 'out%d.%d.pr.png' % (idx, i_inner)))
                # pltimg = plt.imshow(next_fr[i_inner, ...])
                # plt.savefig(os.path.join(opt.save_out, 'out%d.%d.gt.png' % (idx, i_inner)))
            idx += 1
            if idx == 10:
                break


