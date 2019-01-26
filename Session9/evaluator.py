import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import os as os
from arguments import opt
import pdb
from utils import batch_iou, performance_metric, peak_detection, predict_box, performance_metric_alternative, tp_fp_tn_fn
import torch.nn.functional as F
import numpy as np
import h5py

class ModelEvaluator:
    def __init__(self, model, threshold):
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
        self.threshold = threshold
        self.fdr_train = []
        self.RC_train = []
        self.accuracy_train = []
        self.fdr_test = []
        self.RC_test = []
        self.accuracy_test = []        
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
        TP, FP, FN, TN = 0, 0, 0, 0
        for b_idx, (train_data, train_labels, box_actual) in enumerate(trainloader):
            if self.use_gpu:
                train_data = train_data.cuda(non_blocking=True)
                train_labels = train_labels.cuda()
            output = self.model(train_data)

            loss = self.loss(output, train_labels)
            
            if self.l2:
                loss = self.l2_regularization(loss, self.l2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            output = output.cpu().detach().squeeze()
            if len(output.shape)<3:
                output = output.unsqueeze(0)
            peaks_predicted_train, peaks_value_train = peak_detection(self.threshold, output.numpy())
            #TP_t, FP_t, TN_t, FN_t  = eval_alt(peaks_predicted_train, box_actual.numpy())
            TP_t, FP_t, FN_t, TN_t = tp_fp_tn_fn(peaks_predicted_train, peaks_value_train, box_actual.numpy())
            TP += TP_t
            FP += FP_t
            FN += FN_t
            TN += TN_t
            if b_idx % opt.print_every == 0:
                print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\t Loss {4}'.
                      format(epoch, b_idx * len(train_data),
                             len(trainloader.dataset),
                             100. * b_idx / len(trainloader), loss))
            
            loss_ = loss.item()
            self.iter_loss_train.append(loss_)
            loss_batch += loss_
        
        FDR_train, RC_train, accuracy_train = performance_metric_alternative(TP, FP, FN, TN)
        self.fdr_train.append(FDR_train)
        self.accuracy_train.append(accuracy_train)
        self.RC_train.append(RC_train)
        
        loss_batch /= len(trainloader)

        print('Epoch = {} Train TP {} FP {} TN {} FN {} '.format(epoch, TP, FP, TN, FN))
        print('Train loss = {0} FDR = {1:.4f} , RC {2:.4f} =, accuracy = {3:.4f}'.format(loss_batch, FDR_train, RC_train, accuracy_train))
        self.train_loss.append(loss_batch)

    def test(self, epoch, testloader):
        '''
        method for testing
        '''
        self.model.eval()
        TP, FP, FN, TN = 0, 0, 0, 0
        with torch.no_grad():
            batch_loss = 0
            for test_data, test_labels, box_actual in testloader:
                if self.use_gpu:
                    test_data, test_labels = test_data.cuda(), test_labels.cuda()
                output = self.model(test_data)
                loss_ = self.loss(output, test_labels)
                output = output.cpu().squeeze()
                if len(output.shape)<3:
                    output = output.unsqueeze(0)
                peaks_predicted_test, peaks_value_test = peak_detection(self.threshold, output.numpy())
                #box_predicted = predict_box(peaks_predicted, box_actual.numpy())
                #TP_test, FP_test, TN_test, FN_test = eval_alt(peaks_predicted_test, box_actual.numpy())
                TP_test, FP_test, FN_test, TN_test = tp_fp_tn_fn(peaks_predicted_test, peaks_value_test, box_actual.numpy())
                #FDR_batch, RC_batch, accuracy_batch = performance_metric(box_actual.numpy(), box_predicted)
                TP += TP_test
                FP += FP_test
                FN += FN_test
                TN += TN_test
                self.iter_loss_test.append(loss_)
                batch_loss += loss_
            
            batch_loss /= len(testloader)
            FDR_test, RC_test, accuracy_test = performance_metric_alternative(TP, FP, FN, TN)
            
            self.fdr_test.append(FDR_test)
            self.accuracy_test.append(accuracy_test)
            self.RC_test.append(RC_test)

            print('epoch {} Test TP {} FP {} TN {} FN {}'.format(epoch, TP, FP, TN, FN))            
            print('Test loss = {0} FDR = {1:.4f} , RC {2:.4f} =, accuracy = {3:.4f}'.format(batch_loss, FDR_test, RC_test, accuracy_test))

            self.test_loss.append(batch_loss)

    def evaluator(self, trainloader, testloader, print_every=1000):
        '''
        train and validate model
        '''
        resume_epoch = 0
        if self.resume:
            checkpoint, resume_epoch = self.load_model('/Model_lr_{}_opt_{}_epoch_{}_net_{}_drop_{}.pth'.format(self.lr, self.optim, epoch, opt.net, opt.drop_p))
            self.model.load_state_dict(checkpoint)
        print('Model')
        print(self.model)
        for epoch in range(resume_epoch, self.epochs):
            self.train(epoch, trainloader, print_every=print_every)
            self.test(epoch, testloader)
            if epoch%opt.save_every==0:
                save_model = {'threshold': self.threshold,
                                'epoch': epoch, 
                                'state_dict_model': self.model.state_dict()}
                model_name = 'Model_lr_{}_opt_{}_epoch_{}_net_{}_drop_{}'.format(self.lr, self.optim, epoch, opt.net, opt.drop_p)
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
        plt.savefig('{}/loss_evaluation_epoch_{}_drop_{}.png'.format(opt.result_root, opt.net, opt.drop_p))
        plt.cla()
        plt.clf()
        plt.plot(range(len(self.iter_loss_train)), self.iter_loss_train,
                 label='Training Loss')
        plt.plot(range(len(self.iter_loss_test)), self.iter_loss_test,
                     label='Testing Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('{}/loss_evaluation_iter_{}_drop_{}.png'.format(opt.result_root, opt.net, opt.drop_p))
        plt.cla()
        plt.clf()
        plt.plot(range(len(self.fdr_train)), self.fdr_train,
                 label='Training FDR')
        plt.plot(range(len(self.fdr_test)), self.fdr_test,
                     label='Testing FDR')
        plt.xlabel('Epoch')
        plt.ylabel('FDR')
        plt.legend()
        plt.savefig('{}/FDR_evaluation_epoch_{}_drop_{}.png'.format(opt.result_root, opt.net, opt.drop_p))   
        plt.cla()
        plt.clf()
        
        plt.plot(range(len(self.RC_train)), self.RC_train,
                 label='Training RC')
        plt.plot(range(len(self.RC_test)), self.RC_test,
                     label='Testing RC')
        plt.xlabel('Epoch')
        plt.ylabel('RC')
        plt.legend()
        plt.savefig('{}/rc_evaluation_epoch_{}_drop_{}.png'.format(opt.result_root, opt.net, opt.drop_p))
        plt.cla()
        plt.clf()
        plt.plot(range(len(self.accuracy_train)), self.accuracy_train,
                 label='Training Acc')
        plt.plot(range(len(self.accuracy_test)), self.accuracy_test,
                     label='Testing Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('{}/accuracy_evaluation_epoch_{}_drop_{}.png'.format(opt.result_root, opt.net, opt.drop_p))
                 
    def load_model(self, model_name):
        '''
        load model checkpoint
        '''
        model_dir = opt.model_root + '/' + model_name
        checkpoint = torch.load(model_dir)
        model, epoch = checkpoint['state_dict_model'], checkpoint['epoch']
        return model, epoch

    def save_output(self):
        filename = '{}/evaluation_epoch_{}_drop_{}.h5'.format(opt.result_root, opt.net, opt.drop_p)
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('RC_train', data = self.RC_train)
            hf.create_dataset('RC_test', data = self.RC_test)
            hf.create_dataset('fdr_train', data = self.fdr_train)
            hf.create_dataset('fdr_test', data = self.fdr_test)
            hf.create_dataset('accuracy_train', data = self.accuracy_train)
            hf.create_dataset('accuracy_test', data = self.accuracy_test)