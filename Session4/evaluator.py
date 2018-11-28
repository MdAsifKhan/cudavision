#!/usr/bin/env python

"""
"""

__author__ = 'Anna Kukleva'
__date__ = 'November 2018'

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from utils import adjust_lr


class ModelEvaluator:
    def __init__(self, model, epochs, lr, l2=0.0, use_gpu=False, optim='adam'):
        '''
        model: instance of pytorch model class
        epochs: number of training epochs
        lr: learning rate
        use_gpu: to use gpu
        optim: optimizer used for training, SGD or adam
        '''
        self.epochs = epochs
        self.lr = lr
        self.model = model
        self.l2 = l2
        self.use_gpu = use_gpu
        self.train_loss = []
        self.test_loss = []

        if self.use_gpu:
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')

            if str(self.device) == 'cuda:0':
                if torch.cuda.device_count() > 1:
                    self.model = nn.DataParallel(model)
                self.model.to(self.device)

        if optim == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr,
                                             momentum=0.9)
        elif optim == 'adadelta':
            self.optimizer = torch.optim.Adadelta(self.model.parameters(),
                                                  lr=lr, eps=1e-6,
                                                  weight_decay=0)
        elif optim == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=lr,
                                                 lr_decay=1e-6, weight_decay=0)
        elif optim == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr,
                                                 alpha=0.995, eps=1e-7,
                                                 weight_decay=0)
        else:
            ValueError('Optimizer Not Supported')

    def l2_regularization(self, loss, lam):
        l2 = 0
        for W in self.model.parameters():
            l2 += W.norm(2)
        loss = loss + 0.5 * lam * l2
        return loss

    def train(self, epoch, trainloader, print_every=100):
        '''
        method for training
        '''
        loss_batch = 0
        if epoch % 10 == 0 and epoch > 0:
            adjust_lr(self.optimizer, self.lr)
        for b_idx, (train_data, train_labels) in enumerate(trainloader):
            if self.use_gpu and str(self.device) == 'cuda:0':
                train_data = train_data.cuda(non_blocking=True)
                train_labels = train_labels.cuda()

            # Forward Pass
            train_preds = self.model.forward(train_data)
            loss = self.model.loss(train_preds, train_labels)
            if self.l2:
                loss = self.l2_regularization(loss, self.l2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if b_idx % print_every == 0:
                print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\t Loss {4:.6f}'.
                      format(epoch, b_idx * len(train_data),
                             len(trainloader.dataset),
                             100. * b_idx / len(trainloader), loss))

            loss_batch += loss.item()
        loss_batch /= len(trainloader)
        self.train_loss.append(loss_batch)

    def validation(self, valloader):
        '''
        method for validation
        '''
        correct_, total_ = 0, 0
        with torch.no_grad():
            loss = 0
            for val_data, val_labels in valloader:
                if self.use_gpu and str(self.device) == 'cuda:0':
                    val_data, val_labels = val_data.to(
                        self.device), val_labels.to(self.device)
                val_preds = self.model.forward(val_data)

                loss += self.model.loss(val_preds, val_labels)

                _, val_pred_labels = torch.max(val_preds.data, 1)
                total_ += val_labels.size(0)
                correct_ += (val_pred_labels.cpu() == val_labels.cpu()).sum()

            loss /= len(valloader)
            self.val_loss.append(loss)
            accuracy_val = (100.0 * correct_ / total_)
            print(
                'Validation Loss {1:.2f} Accuracy on validation set {2:.2f}'.format(
                    loss, accuracy_val))
            return accuracy_val

    def test(self, testloader):
        '''
        method for testing
        '''
        correct_, total_ = 0, 0
        with torch.no_grad():
            loss = 0
            for test_data, test_labels in testloader:
                if self.use_gpu and str(self.device) == 'cuda:0':
                    test_data, test_labels = test_data.to(
                        self.device), test_labels.to(self.device)
                test_preds = self.model.forward(test_data)

                loss += self.model.loss(test_preds, test_labels)
                _, test_pred_labels = torch.max(test_preds.data, 1)
                total_ += test_labels.size(0)
                correct_ += (test_pred_labels.cpu() == test_labels.cpu()).sum()

            loss /= len(testloader)
            self.test_loss.append(loss)
            accuracy_test = (100 * correct_ / total_)
            print('Accuracy of model on test set {0:.2f}'.format(accuracy_test))
            return accuracy_test

    def evaluator(self, trainloader, testloader, print_every=1000,
                  validation=False):
        for epoch in range(self.epochs):
            self.train(epoch, trainloader, print_every=print_every)
            if validation:
                acc_ = self.validation(testloader)
            else:
                acc_ = self.test(testloader)
        return acc_

    def plot_loss(self, validation=False):
        '''
        to visualize loss
        '''
        plt.plot(range(len(self.train_loss)), self.train_loss,
                 label='Training Loss')
        if validation:
            plt.plot(range(len(self.val_loss)), self.val_loss,
                     label='Validation Loss')
        else:
            plt.plot(range(len(self.test_loss)), self.test_loss,
                     label='Testing Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()