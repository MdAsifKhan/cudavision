import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle as skshuffle
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import pdb
from visualization import heatmap, annotate_heatmap
from matplotlib import pyplot as plt
import copy

class LogisticRegression(torch.nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(LogisticRegression, self).__init__()
        '''
        n_in: Number of Inputs
        n_hidden: Number of Hidden Units
        n_out: Number of Output Units
        '''
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.fc1 = nn.Linear(self.n_in, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden, self.n_out)
        self.nonlin = nn.ReLU()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, X):
        return self.fc2(self.nonlin(self.fc1(X)))

class ModelEvaluator:
    def __init__(self, model, epochs, lr, use_gpu=False, optim='adam'):
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
        self.use_gpu = use_gpu
        self.epoch_loss = []
        if self.use_gpu:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            if self.device == 'cuda:0':
                if torch.cuda.device_count()>1:
                    self.model = nn.DataParallel(model)
                self.model.to(device)
        if optim=='adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optim=='sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr = lr, momentum=0.9)
        else:
            ValueError('Optimizer Not Supported')


    def train(self, trainloader, testloader, validation=False):
        '''
        method for training
        '''
        iter_ = 0
        for epoch in range(self.epochs):
            print('Epoch-{}'.format(epoch+1))
            print('-----------------')
            loss_batch = []
            for train_data, train_labels in trainloader:
                if self.use_gpu and self.device == 'cuda:0':
                    train_data, train_labels = train_data.to(self.device), train_labels.to(self.device)
                train_data = train_data.reshape(-1, 32*32*3)
                train_data = train_data / 255
                train_preds = self.model.forward(train_data)
                loss = self.model.loss(train_preds, train_labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                iter_ += 1
                print('Iter-{0}, training loss{1:.2f}'.format(iter_, loss))
                if not validation:
                    if iter_%500 == 0:
                        acc_test = self.test(validloader)
                        print('Accuracy on Test Set {:.2f}'.format(acc_test))
                loss_batch.append(loss)
            self.epoch_loss.append(np.sum(loss_batch))    

    def test(self, testloader):
        '''
        method for testing
        '''
        correct_ = 0
        total_ = 0
        with torch.no_grad():
            for test_data, test_labels in testloader:
                if self.use_gpu and self.device == 'cuda:0':
                    test_data, test_labels = test_data.to(self.device), test_labels.to(self.device)
                test_data = test_data.reshape(-1, 32*32*3)
                test_data = test_data / 255
                test_preds = self.model.forward(test_data)
                _, test_pred_labels = torch.max(test_preds.data, 1)
                total_ += test_labels.size(0)
                correct_ += (test_pred_labels.cpu() == test_labels.cpu()).sum()
                accuracy_test = (100*correct_/total_)
            return accuracy_test
    
    def plot_loss(self):
        '''
        to visualize loss
        '''
        plt.plot(range(len(self.epoch_loss)), self.epoch_loss)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()


class CrossValidation:
    def __init__(self, k, batch_size, trainset, use_gpu):
        '''
        k: number of folds
        batch_size: batch size for training
        trainset: training data as pytorch iterator
        use_gpu: boolean variable to use gpus
        '''
        self.k = k
        self.nm_samples = len(trainset)
        self.indices = list(range(self.nm_samples))
        self.trainset = trainset
        self.batch_size = batch_size

    def kfold(self):
        '''
        k-fold split
        '''
        for i in range(self.k):
            train_idx = [idx for j,idx in enumerate(self.indices) if j%self.k != i]
            valid_idx = [idx for j,idx in enumerate(self.indices) if j%self.k == i]            
            yield train_idx, valid_idx
    
    def trainloader_sampling(self):
        '''
        k-fold samples
        '''
        for train_idx, valid_idx in self.kfold():
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            yield train_sampler, valid_sampler

    def gridsearchCV(self, parameters):
        '''
        find best parameters by doing grid search with k-fold cross validation
        '''
        accuracy_mat = np.zeros(len(parameters['lr'], len(parameters['n_hidden'])))
        for ii,lr in enumerate(parameters['lr']):
            for jj,n_hidden in enumerate(parameters['n_hidden']):
                fold_accuracy = []
                for i,train_sampler, valid_sampler in enumerate(self.trainloader_sampling()):
                    trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=2)
                    validloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, sampler=valid_sampler, num_workers=2)    
                    
                    model = LogisticRegression(n_in, n_hidden, n_out)
                    modeleval = ModelEvaluator(model, epochs, lr, use_gpu=self.use_gpu)
                    modeleval.train(trainloader, validloader, validation=True)
                    #modeleval.plot_loss()
                    accuracy_valid = modeleval.test(validloader)
                    print('Accuracy of model on validation set {0:.2f}'.format(accuracy_valid))
                    fold_accuracy.append(accuracy_valid)
                mean_acc = np.mean(fold_accuracy)
                if mean_acc > np.max(accuracy_mat):
                    best_model, best_lr, best_n_hidden = copy.deepcopy(model), lr, n_hidden
                    bestmodeleval = copy.deepcopy(modeleval)
                accuracy_mat[ii, jj] = np.mean(fold_accuracy)
        return accuracy_mat, best_model, best_lr, best_n_hidden, bestmodeleval

if __name__ == '__main__':
    
    trainset = dsets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    testset = dsets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    n_in = np.prod(trainset[0][0].numpy().shape)
    n_out = len(classes)
    batch_size = 100
    epochs = 100

    # Number of Parameters
    parameters = {'lr':[0.00001, 0.0001, 0.001, 0.01], 'n_hidden': [512, 256, 128]}
    # k fold cross validation
    k = 3
    cv = CrossValidation(k=k, batch_size=batch_size, trainset=trainset, use_gpu=True)
    accuracy_mat, best_model, best_lr, best_n_hidden, bestmodeleval = cv.gridsearchCV(parameters)

    # Visualization accuracy vs parameters
    fig, ax = plt.subplots()
    lr_ = [str(lr) for lr in parameters['lr']]
    hidden_ = [str(n_hidden) for n_hidden in parameters['n_hidden']]
    im, cbar = heatmap(accuracy_mat, lr_, hidden_, ax=ax,
                       cmap='YlGn', cbarlabel='lr vs hidden_')
    texts = annotate_heatmap(im, valfmt='{x:.1f} t')
    fig.tight_layout()
    plt.show()
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    accuracy_test = bestmodeleval.test(testloader)
    print('Accuracy of best model on test set with lr= {0:.2f}, hidden units= {1:.2f}, is {2:.2f}'.format(best_lr, best_n_hidden, accuracy_test))