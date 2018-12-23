from model import LSTMModel, GRUModel
from evaluator import ModelEvaluator
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np




trainset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
testset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())


batch_size = 100
 
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)


n_in = 28
n_hidden = 100
n_out = 10
seq_dim = 28
use_gpu = True

#model = LSTMModel(n_in, n_hidden, n_out, batch_size, use_gpu)
model = GRUModel(n_in, n_hidden, n_out, batch_size, use_gpu)

if use_gpu:
    model.cuda()

l2 = 0.0
lr = 0.001
epochs = 10
optim = 'adam'
modeleval = ModelEvaluator(model, epochs, lr, batch_size, l2, use_gpu, optim)
acc_ = modeleval.evaluator(trainloader, testloader, seq_dim, n_in)
