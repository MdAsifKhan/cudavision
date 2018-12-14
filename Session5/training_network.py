#!/usr/bin/env python

"""Implementation of training and testing functions for embedding."""


import torch
import torch.backends.cudnn as cudnn
import os.path as ops
import time
import numpy as np

from utils.arg_pars import opt
from utils.logging_setup import logger
from utils.utils import Averaging, adjust_lr
from utils.utils import dir_check


def training(dataloader, **kwargs):
    """Training pipeline for embedding.

    Args:
        dataloader: iterator within dataset
        epochs: how much training epochs to perform
        n_subact: number of subactions in current complex activity
        mnist: if training with mnist dataset (just to test everything how well
            it works)
    Returns:
        trained pytorch model
    """
    logger.debug('create model')
    torch.manual_seed(opt.seed)

    model = kwargs['model']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']

    cudnn.benchmark = True

    batch_time = Averaging()
    data_time = Averaging()
    losses = Averaging()

    adjustable_lr = opt.lr

    logger.debug('epochs: %s', opt.epochs)
    for epoch in range(opt.epochs):
        model.train()

        logger.debug('Epoch # %d' % epoch)
        if opt.lr_adj:
            if epoch % 5 == 0 and epoch > 0:
                adjustable_lr = adjust_lr(optimizer, adjustable_lr)
                logger.debug('lr: %f' % adjustable_lr)
        end = time.time()
        for i, (features, labels) in enumerate(dataloader):
            data_time.update(time.time() - end)
            features = features.cuda(non_blocking=True)
            # features = features.float().cuda(non_blocking=True)
            labels = labels.long().cuda()
            output = model(features)
            loss_values = loss(output, labels)
            losses.update(loss_values.item(), labels.size(0))

            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0 and i:
                logger.debug('Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(dataloader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
        logger.debug('loss: %f' % losses.avg)
        losses.reset()

        if epoch % 1 == 0:
            save_dict = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
            dir_check(ops.join(opt.dataset_root, 'models'))
            torch.save(save_dict, ops.join(opt.dataset_root, 'models', '%s%d.pth.tar' % (opt.log_str, epoch)))

    if opt.save_model:
        save_dict = {'epoch': opt.epochs,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        dir_check(ops.join(opt.dataset_root, 'models'))
        torch.save(save_dict, ops.join(opt.dataset_root, 'models', '%s%d.pth.tar' % (opt.log_str, opt.epochs)))
    return model


def load_model(epoch=None):
    if opt.resume_str:
        resume_str = opt.resume_str
    else:
        resume_str = opt.log_str
    epoch = opt.epochs if epoch is None else epoch
    checkpoint = torch.load(ops.join(opt.dataset_root, 'models', '%s%d.pth.tar' % (resume_str, epoch)))
    checkpoint = checkpoint['state_dict']
    logger.debug('loaded model: ' + '%s%d.pth.tar' % (resume_str, opt.epochs))
    return checkpoint


