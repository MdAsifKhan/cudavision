import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle as skshuffle
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import pdb

class MLP(torch.nn.Module):
	def __init__(self, n_in, n_hidden, drop_p, n_out):
		super(MLP, self).__init__()
		'''
		n_in: Number of Inputs
		n_hidden: List with units in hidden layers
		n_out: Number of Output Units
		'''
		self.n_in = n_in
		self.n_out = n_out
		self.n_hidden = n_hidden
		self.p = drop_p
		self.input_layer = nn.Linear(self.n_in, self.n_hidden[0])
		self.input_nonlin = nn.ReLU()
		self.hidden = nn.ModuleList()

		for i in range(len(self.n_hidden)-1):
			self.hidden.append(nn.Linear(self.n_hidden[i], self.n_hidden[i+1]))
			self.hidden.append(nn.ReLU()) 
			self.hidden.append(nn.Dropout(p = self.p))
		self.final_fc = nn.Linear(self.n_hidden[-1], self.n_out)


	def forward(self, X):
		'''
		forward pass
		'''
		X = self.input_nonlin(self.input_layer(X))
		for layer in self.hidden:
			X = layer(X)
		return X


class ModelEvaluator:
	def __init__(self, model, epochs, lr, loss_type='softmax', l2=0.0, use_gpu=False, optim='adam'):
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
		self.loss_type = loss_type
		self.l2 = l2
		self.use_gpu = use_gpu
		self.train_loss = []
		self.test_loss = []

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

	def softmax_loss(self, scores, y_label):
		'''
		softmax loss without torch.nn
		'''
		exp_scores = torch.exp(scores - torch.max(scores, dim=1)[0].view(-1,1))
		sum_exp_scores = exp_scores.sum(dim=1).view(-1,1)
		softmax_prob = exp_scores/sum_exp_scores
		loss = -torch.log(softmax_prob[np.arange(len(scores)), y_label])
		loss = loss.sum()
		return loss

	def hinge_loss(self, scores, y_label):
		'''
		hinge loss for multiclass without torch.nn
		'''
		hinge_loss = torch.clamp(scores - scores[np.arange(len(scores)),y_label].view(-1,1) + 1, min=0)
		hinge_loss[np.arange(len(scores)),y_label] = 0
		hinge_loss = hinge_loss.sum()

		return loss

	def l2_regularization(self, loss, lam):
		l2 = 0
		for W in self.model.parameters():
			l2 += W.norm(2)
		loss = model.loss() + 0.5*lam*l2
		return loss	


	def train(self, epoch, trainloader, print_every=100):
		'''
		method for training
		'''
		loss_batch = 0
		for b_idx, (train_data, train_labels) in enumerate(trainloader):
			if self.use_gpu and self.device == 'cuda:0':
				train_data, train_labels = train_data.to(self.device), train_labels.to(self.device)
			train_data = train_data.reshape(-1, 32*32*3)
			
			# Scale Images
			train_data = train_data / 255
			
			# Forward Pass 
			train_preds = self.model.forward(train_data)
			
			if self.loss_type=='softmax':
				loss = self.softmax_loss(train_preds, train_labels)
			elif self.loss_type =='hinge':
				loss = self.hinge_loss(train_preds, train_labels)
			else:
				ValueError('Loss Not Supported')
			if self.l2:
				loss = l2_regularization(loss, lam)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			
			if b_idx%print_every == 0:
				print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\t Loss {4:.6f}'.
					format(epoch, b_idx*len(train_data), len(trainloader.dataset), 
						100.*b_idx/len(trainloader), loss))
			
			loss_batch += loss
		loss_batch /= len(trainloader)
		self.train_loss.append(loss_batch)    

	def validation(self, valloader):
		'''
		method for testing
		'''
		correct_, total_ = 0, 0
		with torch.no_grad():
			loss = 0
			for val_data, val_labels in valloader:
				if self.use_gpu and self.device == 'cuda:0':
					val_data, val_labels = test_data.to(self.device), val_labels.to(self.device)
				val_data = val_data.reshape(-1, 32*32*3)
				val_data = val_data / 255
				val_preds = self.model.forward(val_data)
				
				if self.loss_type=='softmax':
					loss += self.softmax_loss(val_preds, val_labels)
				elif self.loss_type =='hinge':
					loss += self.hinge_loss(val_preds, val_labels)
				else:
					ValueError('Loss Not Supported')
				
				_, val_pred_labels = torch.max(val_preds.data, 1)
				total_ += val_labels.size(0)
				correct_ += (val_pred_labels.cpu() == val_labels.cpu()).sum()
			
			loss /= len(valloader)
			self.val_loss.append(loss)
			accuracy_val = (100.0*correct_/total_)
			print('Validation Loss {1:.2f} Accuracy on validation set {2:.2f}'.format(loss, accuracy_val))
			return accuracy_val

	def test(self, testloader):
		'''
		method for testing
		'''
		correct_, total_ = 0, 0
		with torch.no_grad():
			loss = 0
			for test_data, test_labels in testloader:
				if self.use_gpu and self.device == 'cuda:0':
					test_data, test_labels = test_data.to(self.device), test_labels.to(self.device)
				test_data = test_data.reshape(-1, 32*32*3)
				test_data = test_data / 255
				test_preds = self.model.forward(test_data)
				
				if self.loss_type=='softmax':
					loss += self.softmax_loss(test_preds, test_labels)
				elif self.loss_type =='hinge':
					loss += self.hinge_loss(test_preds, test_labels)
				else:
					ValueError('Loss Not Supported')
				
				_, test_pred_labels = torch.max(test_preds.data, 1)
				total_ += test_labels.size(0)
				correct_ += (test_pred_labels.cpu() == test_labels.cpu()).sum()
			
			loss /= len(testloader)
			self.test_loss.append(loss)
			accuracy_test = (100*correct_/total_)
			print('Accuracy of model on test set {0:.2f}'.format(accuracy_test))
			return accuracy_test
	
	def evaluator(self, trainloader, testloader, print_every=1000, validation=False):
		for epoch in range(self.epochs):
			self.train(epoch, trainloader, print_every=print_every)
			if validation:
				self.validation(testloader)
			else:
				self.test(testloader)

	def plot_loss(self, validation=False):
		'''
		to visualize loss
		'''
		plt.plot(range(len(self.train_loss)), self.train_loss, label='Training Loss')
		if validation:
			plt.plot(range(len(self.val_loss)), self.val_loss, label='Testing Loss')
		else:
			plt.plot(range(len(self.test_loss)), self.test_loss, label='Testing Loss')
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
	epochs = 20
	# Data Loader
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

	# Model
	n_hidden = [512, 256, 128]
	l2 = False
	drop_p = 0.3
	loss_type = 'softmax'
	model = MLP(n_in, n_hidden, drop_p, n_out)
	modeleval = ModelEvaluator(model, epochs, lr, loss_type=loss_type, l2=l2, use_gpu=True)
	modeleval.evaluator(trainloader, testloader, print_every=100, validation=False)
	modeleval.plot_loss()
	#accuracy_test = modeleval.test(testloader)
	#print('Accuracy of model on test set {0:.2f}'.format(accuracy_test))