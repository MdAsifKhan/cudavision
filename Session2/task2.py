import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle as skshuffle
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

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
		'''
		forward pass
		'''
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
				if validation:
					if iter_%500 == 0:
						acc_test = self.test(testloader)
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

if __name__ == '__main__':

	trainset = dsets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
	testset = dsets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	# Parameters
	n_in = np.prod(trainset[0][0].numpy().shape)
	n_out = len(classes)
	batch_size = 100

	# Hyperparameters
	lr = 0.001
	n_hidden = 512
	epochs = 100

	# Data Loader
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

	# Model
	model = LogisticRegression(n_in, n_hidden, n_out)
	modeleval = ModelEvaluator(model, epochs, lr, use_gpu=True)
	modeleval.train(trainloader, testloader)
	modeleval.plot_loss()
	accuracy_test = modeleval.test(testloader)
	print('Accuracy of model on test set {0:.2f}'.format(accuracy_test))