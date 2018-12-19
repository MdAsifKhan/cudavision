#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'November 2018'

def adjust_lr(optimizer, lr):
    """Decrease learning rate by 0.1 during training"""
    lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr