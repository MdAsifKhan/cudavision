#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'November 2018'

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, pool='max'):
        super(CNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                              padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                              padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.cnn3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                              padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.cnn4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                              padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.cnn5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                              padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        if pool == 'max':
            self.pool = nn.MaxPool2d(kernel_size=2)

        else:
            self.pool = nn.AvgPool2d(kernel_size=2)

        self.drop = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        # self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.pool(self.drop(self.relu1(self.bn1(self.cnn1(x)))))

        out = self.pool(self.drop(self.relu2(self.bn2(self.cnn2(out)))))

        out = self.drop(self.relu3(self.bn3(self.cnn3(out))))

        out = self.pool(self.drop(self.relu4(self.bn4(self.cnn4(out)))))

        out = self.pool(self.drop(self.relu5(self.bn5(self.cnn5(out)))))

        out = out.view(out.size(0), -1)

        out = self.fc2(self.drop(self.relu6(self.fc1(out))))

        return out

    def cnn1_out(self, x):
        return self.cnn1(x)

    def cnn2_out(self, x):
        out = self.pool(self.drop(self.relu1(self.bn1(self.cnn1(x)))))
        return self.cnn2(out)

    def cnn3_out(self, x):
        out = self.cnn2_out(x)
        out = self.pool(self.drop(self.relu2(self.bn2(out))))
        return self.cnn3(out)

    def cnn4_out(self, x):
        out = self.cnn3_out(x)
        out = self.pool(self.drop(self.relu3(self.bn3(out))))
        return self.cnn4(out)