import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import os as os

class AutoEncoderEvaluator:
    def __init__(self, encoder, decoder, epochs, lr, batch_size, l2=0.0 , add_noise=False, use_gpu=False, optim='adam'):
        '''
        model: instance of pytorch model class
        epochs: number of training epochs
        lr: learning rate
        use_gpu: to use gpu
        optim: optimizer used for training, SGD or adam
        '''
        self.epochs = epochs
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.l2 = l2
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.train_loss = []
        self.test_loss = []
        self.train_loss_clf = []
        self.test_loss_clf = []
        self.accuracy_test_clf = []
        self.add_noise = add_noise
        self.loss = nn.MSELoss()
        self.optim = optim
        if self.use_gpu:
            self.device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')

            if str(self.device) == 'cuda:0':
                if torch.cuda.device_count() > 1:
                    self.encoder = nn.DataParallel(encoder)
                    self.decoder = nn.DataParallel(decoder)
                self.encoder.to(self.device)
                self.decoder.to(self.device)

        parameters = list(self.encoder.parameters())+ list(self.decoder.parameters())
        if optim == 'adam':
            self.optimizer = torch.optim.Adam(parameters, lr=lr)
        elif optim == 'sgd':
            self.optimizer = torch.optim.SGD(parameters, lr=lr,
                                             momentum=0.9)
        elif optim == 'adadelta':
            self.optimizer = torch.optim.Adadelta(parameters,
                                                  lr=lr, eps=1e-6,
                                                  weight_decay=0)
        elif optim == 'adagrad':
            self.optimizer = torch.optim.Adagrad(parameters, lr=lr,
                                                 lr_decay=1e-6, weight_decay=0)
        elif optim == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(parameters, lr=lr,
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
    
    def adjust_lr(self, step=0.1):
        'Decrease learning rate by 0.1 during training'
        lr = self.lr * step
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train(self, epoch, trainloader, print_every=100):
        '''
        method for training
        '''

        loss_batch = 0
        if epoch % 10 == 0 and epoch > 0:
            self.adjust_lr(step=0.1)
        for b_idx, (train_data, train_labels) in enumerate(trainloader):
            if self.add_noise:
                train_data_n = torch.mul(train_data+0.25, 0.1*torch.rand(batch_size, 3, 32, 32))
            
            if self.use_gpu and str(self.device) == 'cuda:0':
                train_data = train_data.cuda(non_blocking=True)
                train_labels = train_labels.cuda()

            # Forward Pass
            if self.add_noise:
                latent_repr = self.encoder.forward(train_data_n)
            else:
                latent_repr = self.encoder.forward(train_data)
            
            output = self.decoder.forward(latent_repr)
            loss = self.loss(output, train_data)
            
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

                save_model = {'epoch': epoch, 
                                'state_dict_encoder': self.encoder.state_dict(),
                                'state_dict_decoder': self.decoder.state_dict(), 
                                    'optimizer': self.optimizer.state_dict()}
                
                model_name = 'AutoEncoder_lr_{}_opt_{}_epoch_{}'.format(self.lr, self.optim, epoch)
                model_dir = '../Session6/model/' + model_name
                
                if self.add_noise:
                    model_dir = model_dir + '_dae'
                torch.save(save_model, model_dir)
            loss_batch += loss.item()
        loss_batch /= len(trainloader)
        self.train_loss.append(loss_batch)

    def test(self, testloader):
        '''
        method for testing
        '''
        correct_, total_ = 0, 0
        with torch.no_grad():
            loss = 0
            for test_data, test_labels in testloader:
                if self.add_noise:
                    test_data_n = torch.mul(test_data+0.25, 0.1 * torch.rand(self.batch_size, 3, 32, 32))
                if self.use_gpu and str(self.device) == 'cuda:0':
                    test_data, test_labels = test_data.to(
                        self.device), test_labels.to(self.device)

                if self.add_noise:
                    latent_repr = self.encoder.forward(test_data_n)
                else:
                    latent_repr = self.encoder.forward(test_data)
            
                output = self.decoder.forward(latent_repr)
                loss += self.loss(output, test_data)

            loss /= len(testloader)
            self.test_loss.append(loss)

    def evaluator(self, trainloader, testloader, print_every=1000):
        for epoch in range(self.epochs):
            self.train(epoch, trainloader, print_every=print_every)
            self.test(testloader)

    def plot_loss(self):
        '''
        to visualize loss
        '''
        plt.plot(range(len(self.train_loss)), self.train_loss,
                 label='Training Loss')
        plt.plot(range(len(self.test_loss)), self.test_loss,
                     label='Testing Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()



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
        self.optim = optim

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
        loss = loss + 0.5 * lam * l2**2
        return loss

    def train(self, encoder, epoch, trainloader, noise=False, print_every=100):
        '''
        method for training
        '''
        model.train()
        loss_batch = 0
        if epoch % 10 == 0 and epoch > 0:
            adjust_lr(step=0.1)
        for b_idx, (train_data, train_labels) in enumerate(trainloader):
            if noise:
                    train_data_n = torch.mul(train_data+0.25, 0.1 * torch.rand(self.batch_size, 3, 32, 32))
            if self.use_gpu and str(self.device) == 'cuda:0':
                train_data = train_data.cuda(non_blocking=True)
                train_labels = train_labels.cuda()

            if noise:
                latent_repr = encoder.forward(train_data_n)
            else:
                latent_repr = encoder.forward(train_data)
            
            # Forward Pass
            train_preds = self.model.forward(latent_repr)
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
                
                save_model = {'epoch': epoch, 
                                'state_dict': self.model.state_dict(), 
                                    'optimizer': self.optimizer.state_dict()}

                model_name = 'model_lr_{}_opt_{}_epoch_{}'.format(self.lr, self.optim, epoch)
                model_dir = '../Session6/model/' + model_name
                if noise:
                    model_dir = model_dir + '_dae'
                torch.save(save_model, model_dir)

            loss_batch += loss.item()
        loss_batch /= len(trainloader)
        self.train_loss.append(loss_batch)
        

    def evaluator(self, encoder, trainloader, testloader, noise=False, print_every=1000):
        for epoch in range(self.epochs):
            self.train(encoder, epoch, trainloader, noise=noise, print_every=print_every)
            acc_ = self.test(encoder, testloader)
        return acc_

    def test(self, encoder, testloader):
        '''
        method for testing
        '''
        self.model.eval()
        correct_, total_ = 0, 0
        with torch.no_grad():
            loss = 0
            for test_data, test_labels in testloader:
                if noise:
                    test_data_n = torch.mul(test_data+0.25, 0.1 * torch.rand(self.batch_size, 3, 32, 32))
                if self.use_gpu and str(self.device) == 'cuda:0':
                    test_data, test_labels = test_data.to(
                        self.device), test_labels.to(self.device)

                if noise:
                    latent_repr = encoder.forward(test_data_n)
                else:
                    latent_repr = encoder.forward(test_data)
                
                test_preds = self.model.forward(latent_repr)
                loss += self.model.loss(test_preds, test_labels)

                _, test_pred_labels = torch.max(test_preds.data, 1)
                total_ += test_labels.size(0)
                correct_ += (test_pred_labels.cpu() == test_labels.cpu()).sum()

            loss /= len(testloader)
            self.test_loss.append(loss)
            accuracy_test = (100 * correct_ / total_)
            self.accuracy_test.append(accuracy_test)
            print('Accuracy of model on test set {0:.2f}'.format(accuracy_test))
        return accuracy_test


    def plot_loss(self):
        '''
        to visualize loss
        '''
        plt.plot(range(len(self.train_loss)), self.train_loss,
                 label='Training Loss')
        plt.plot(range(len(self.test_loss)), self.test_loss,
                     label='Testing Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()    