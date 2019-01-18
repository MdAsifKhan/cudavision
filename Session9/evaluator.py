import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import os as os
from arguments import opt
import pdb
from utils import batch_iou, performance_metric, peak_detection, predict_box
import torch.nn.functional as F
import numpy as np

class ModelEvaluator:
    def __init__(self, model):
        '''
        model: instance of pytorch model class
        epochs: number of training epochs
        lr: learning rate
        use_gpu: to use gpu
        optim: optimizer used for training, SGD or adam
        '''
        self.model = model
        self.epochs = opt.nm_epochs
        self.lr = opt.lr
        self.l2 = opt.l2
        self.use_gpu = opt.use_gpu
        self.batch_size = opt.batch_size
        self.train_loss = []
        self.test_loss = []
        self.iter_loss_train = []
        self.iter_loss_test = []
        self.loss = nn.MSELoss()
        self.optim = opt.optimizer
        self.resume = opt.resume
        self.threshold = 0
        if self.use_gpu:
            self.model.cuda()

        parameters = self.model.parameters()
        if self.optim == 'adam':
            self.optimizer = torch.optim.Adam(parameters, lr=opt.lr)
        elif self.optim == 'sgd':
            self.optimizer = torch.optim.SGD(parameters, lr=opt.lr,
                                             momentum=opt.mom)
        elif self.optim == 'adadelta':
            self.optimizer = torch.optim.Adadelta(parameters,
                                                  lr=opt.lr, eps=opt.eps,
                                                  weight_decay=opt.decay)
        elif self.optim == 'adagrad':
            self.optimizer = torch.optim.Adagrad(parameters, lr=lr,
                                                 lr_decay=opt.lr_decay, weight_decay=opt.decay)
        elif self.optim == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(parameters, lr=lr,
                                                 alpha=opt.alpha, eps=opt.eps,
                                                 weight_decay=opt.decay)
        else:
            ValueError('Optimizer Not Supported')

    def l2_regularization(self, loss, lam):
        '''
        regularize output
        '''
        l2 = 0.0
        for W in self.model.parameters():
            l2 += W.norm(2)
        loss = loss + 0.5 * lam * l2**2
        return loss
    
    def adjust_lr(self, step=0.1):
        '''
        Decrease learning rate by 0.1 during training
        '''
        lr = self.lr * step
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train(self, epoch, trainloader, print_every=100):
        '''
        method for training
        '''
        self.model.train()
        loss_batch = 0
        if epoch % 100 == 0 and epoch > 0:
            self.adjust_lr(step=0.1)
        for b_idx, (train_data, train_labels, _) in enumerate(trainloader):
            if self.use_gpu:
                train_data = train_data.cuda(non_blocking=True)
                train_labels = train_labels.cuda()
            output = self.model(train_data)
            threshold = 0.7*train_labels.max()
            if threshold>self.threshold:
                self.threshold = threshold
            loss = self.loss(output, train_labels)
            
            if self.l2:
                loss = self.l2_regularization(loss, self.l2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if b_idx % opt.print_every == 0:
                print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\t Loss {4:.6f}'.
                      format(epoch, b_idx * len(train_data),
                             len(trainloader.dataset),
                             100. * b_idx / len(trainloader), loss))
            
            loss_ = loss.item()
            self.iter_loss_train.append(loss_)
            loss_batch += loss_
        loss_batch /= len(trainloader)
        self.train_loss.append(loss_batch)

    def test(self, epoch, testloader):
        '''
        method for testing
        '''
        self.model.eval()
        FDR, RC, accuracy = 0, 0, 0
        with torch.no_grad():
            batch_loss = 0
            for test_data, test_labels, box_actual in testloader:
                if self.use_gpu:
                    test_data, test_labels = test_data.cuda(), test_labels.cuda()
                output = self.model(test_data)
                loss_ = self.loss(output, test_labels)
                peaks_predicted = peak_detection(threshold, output.cpu().numpy().squeeze())
                box_predicted = predict_box(peaks_predicted, box_actual.numpy())
                
                FDR_batch, RC_batch, accuracy_batch = performance_metric(box_actual.numpy(), box_predicted)
                FDR += FDR_batch
                RC += RC_batch
                accuracy += accuracy_batch
                self.iter_loss_test.append(loss_)
                batch_loss += loss_
            batch_loss /= len(testloader)
            print('Epoch = {0:} loss = {1:.4f} FDR = {2:.4f} , RC {3:.4f} =, accuracy = {4:.4f}'.format(epoch, batch_loss, FDR/len(testloader), RC/len(testloader), accuracy/len(testloader)))

            self.test_loss.append(batch_loss)

    def evaluator(self, trainloader, testloader, print_every=1000):
        '''
        train and validate model
        '''
        resume_epoch = 0
        if self.resume:
            checkpoint, resume_epoch = self.load_model('/Model_lr_{}_opt_{}_epoch_{}.pth'.format(self.lr, self.optim, epoch))
            self.model.load_state_dict(checkpoint)
        print('Model')
        print(self.model)
        for epoch in range(resume_epoch, self.epochs):
            self.train(epoch, trainloader, print_every=print_every)
            if epoch%opt.save_every==0:
                self.test(epoch, testloader)
                save_model = {'threshold': self.threshold,
                                'epoch': epoch, 
                                'state_dict_model': self.model.state_dict()}
                model_name = 'Model_lr_{}_opt_{}_epoch_{}'.format(self.lr, self.optim, epoch)
                model_dir = opt.model_root + '/' + model_name
                torch.save(save_model, model_dir)

    def plot_loss(self):
        '''
        to visualize loss
        '''
        plt.plot(range(len(self.train_loss)), self.train_loss,
                 label='Training Loss')
        plt.plot(range(len(self.test_loss)), self.test_loss,
                     label='Testing Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('{}/loss_evaluation_epoch'.format(opt.result_root))
        plt.cla()
        plt.clf()
        plt.plot(range(len(self.iter_loss_train)), self.iter_loss_train,
                 label='Training Loss')
        plt.plot(range(len(self.iter_loss_test)), self.iter_loss_test,
                     label='Testing Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('{}/loss_evaluation_iter'.format(opt.result_root))
    
    def load_model(self, model_name):
        '''
        load model checkpoint
        '''
        model_dir = opt.model_root + '/' + model_name
        checkpoint = torch.load(model_dir)
        model, epoch = checkpoint['state_dict_model'], checkpoint['epoch']
        return model, epoch