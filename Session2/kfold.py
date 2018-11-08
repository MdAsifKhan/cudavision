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
from task2 import LogisticRegression, ModelEvaluator


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
        self.use_gpu = use_gpu

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
        accuracy_mat = np.zeros((len(parameters['lr']), len(parameters['n_hidden'])))
        for ii,lr in enumerate(parameters['lr']):
            for jj,n_hidden in enumerate(parameters['n_hidden']):
                fold_accuracy = []
                i = 0
                for train_sampler, valid_sampler in self.trainloader_sampling():
                    trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=2)
                    validloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, sampler=valid_sampler, num_workers=2)    
                    
                    model = LogisticRegression(n_in, n_hidden, n_out)
                    modeleval = ModelEvaluator(model, epochs, lr, use_gpu=self.use_gpu)
                    modeleval.train(trainloader, validloader, validation=True)
                    #modeleval.plot_loss()
                    accuracy_valid = modeleval.test(validloader)
                    print('Accuracy of model on validation set {0:.2f}'.format(accuracy_valid))
                    fold_accuracy.append(accuracy_valid)
                    i += 1
                mean_acc = np.mean(fold_accuracy)
                if mean_acc > np.max(accuracy_mat):
                    best_model, best_lr, best_n_hidden = copy.deepcopy(model), lr, n_hidden
                    # bestmodeleval = copy.deepcopy(modeleval)
                accuracy_mat[ii, jj] = np.mean(fold_accuracy)
        return accuracy_mat, best_model, best_lr, best_n_hidden

if __name__ == '__main__':
    
    trainset = dsets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
    testset = dsets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    n_in = np.prod(trainset[0][0].numpy().shape)
    n_out = len(classes)
    batch_size = 100
    epochs = 1

    # Number of Parameters
    parameters = {'lr':[0.00001, 0.0001, 0.001, 0.01], 'n_hidden': [512, 256, 128]}
    # k fold cross validation
    k = 3
    cv = CrossValidation(k=k, batch_size=batch_size, trainset=trainset, use_gpu=True)
    accuracy_mat, best_model, best_lr, best_n_hidden = cv.gridsearchCV(parameters)
    bestmodeleval = ModelEvaluator(best_model, epochs, best_lr, use_gpu=self.use_gpu)
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
    bestmodeleval.train(trainloader, testloader, validation=False)
    accuracy_test = bestmodeleval.test(testloader)
    print('Accuracy of best model on test set with lr= {0:.2f}, hidden units= {1:.2f}, is {2:.2f}'.format(best_lr, best_n_hidden, accuracy_test))