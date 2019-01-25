#!/usr/bin/env python

"""Implementation of training and testing functions for embedding."""

__all__ = ['training', 'load_model']
__author__ = 'Anna Kukleva'
__date__ = 'August 2018'

import torch
import torch.backends.cudnn as cudnn
from os.path import join
import time
import numpy as np
import random

from util_functions import Averaging, adjust_lr, dir_check
from arguments import opt
from logging_setup import logger

def training(train_loader, epochs, save, **kwargs):
    """Training pipeline for embedding.

    Args:
        train_loader: iterator within dataset
        epochs: how much training epochs to perform
    Returns:
        trained pytorch model
    """
    logger.debug('create model')

    # make everything deterministic -> seed setup
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    model = kwargs['model']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']

    cudnn.benchmark = True

    batch_time = Averaging()
    data_time = Averaging()
    losses = Averaging()

    adjustable_lr = opt.lr

    logger.debug('epochs: %s', epochs)
    for epoch in range(epochs):
        model.cuda()
        model.train()

        logger.debug('Epoch # %d' % epoch)
        if opt.lr_adj:
            # if epoch in [int(epochs * 0.3), int(epochs * 0.7)]:
            # if epoch in [int(epochs * 0.5)]:
            if epoch % 30 == 0 and epoch > 0:
                adjustable_lr = adjust_lr(optimizer, adjustable_lr)
                logger.debug('lr: %f' % adjustable_lr)
        end = time.time()
        for i, (features, target) in enumerate(train_loader):
            data_time.update(time.time() - end)
            features = features.float().cuda(non_blocking=True)
            target = target.float().cuda()
            if opt.model == 'lstm':
                outputs, (h, cc) = model(features)
                loss_values = loss(outputs[0], target)
            if opt.model == 'tcn':
                output = model(features)
                loss_values = loss(output, target)
            losses.update(loss_values.item(), features.size(0))

            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0 and i:
                logger.debug('Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
            # logger.debug('output : %s' % str(output))
            # logger.debug('target: %s' % str(target))
        logger.debug('loss: %f' % losses.avg)
        losses.reset()

        if save and opt.fr_save:
            if epoch % opt.fr_save == 0 and epoch:
                save_dict = {'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}
                dir_check(join(opt.dataset_root, 'models'))
                dir_check(join(opt.dataset_root, 'models', kwargs['name']))
                torch.save(save_dict, join(opt.dataset_root, 'models', kwargs['name'],
                                           '%s_%d.pth.tar' % (opt.log_str, epoch)))

    if save:
        save_dict = {'epoch': epoch,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        dir_check(join(opt.dataset_root, 'models'))
        dir_check(join(opt.dataset_root, 'models', kwargs['name']))
        torch.save(save_dict, join(opt.dataset_root, 'models', kwargs['name'],
                                   '%s.pth.tar' % opt.log_str))
    return model


def load_model(name='lstm'):
    checkpoint = torch.load(join(opt.dataset_root, 'models', name,
                                 '%s.pth.tar' % opt.log_str))
    checkpoint = checkpoint['state_dict']
    logger.debug('loaded model: ' + '%s.pth.tar' % opt.log_str)
    return checkpoint

