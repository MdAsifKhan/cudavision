import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import pdb

class ModelEvaluator:
    def __init__(self, model, epochs, lr, batch_size, l2=0.0, use_gpu=False, optim='adam'):
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
        self.batch_size = batch_size
        if self.use_gpu:
            self.model.cuda()

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
        loss = loss + 0.5 * lam * l2**2
        return loss

    def train(self, epoch, trainloader, print_every=100):
        '''
        method for training
        '''
        self.model.train()
        loss_batch = 0
        if epoch % 10 == 0 and epoch > 0:
            self.adjust_lr(self.optimizer, self.lr)
        for b_idx, (train_data, train_labels) in enumerate(trainloader):
            train_data = train_data.view(-1, self.seq_dim, self.n_in)
            if self.use_gpu:
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


    def test(self, testloader):
        '''
        method for testing
        '''
        self.model.eval()
        correct_, total_ = 0, 0
        with torch.no_grad():
            loss = 0
            for test_data, test_labels in testloader:
                test_data = test_data.view(-1, self.seq_dim, self.n_in)
                if self.use_gpu:
                    test_data, test_labels = test_data.cuda(), test_labels.cuda()
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

    def evaluator(self, trainloader, testloader, seq_dim, n_in, print_every=100):
        self.seq_dim = seq_dim
        self.n_in = n_in
        for epoch in range(self.epochs):
            self.train(epoch, trainloader, print_every=print_every)
            acc_ = self.test(testloader)
        return acc_

    def plot_loss(self):
        '''
        to visualize loss
        '''
        plt.plot(range(len(self.train_loss)), self.train_loss,
                 label='Training Loss')
        plt.plot(range(len(self.test_loss)), self.test_loss,
                     label='Testing Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def adjust_lr(self, step=0.1):
        'Decrease learning rate by 0.1 during training'
        lr = self.lr * step
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr