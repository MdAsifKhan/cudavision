
import torch
import torch.nn as nn
from torch.nn import Parameter
import math
import pdb

class GRUCell(nn.Module):
	def __init__(self, n_in, n_hidden):
		super(GRUCell, self).__init__()
		'''
		n_in: Number of Input Units
		n_hidden: Number of Hidden Units
		'''
		self.n_in = n_in
		self.n_hidden = n_hidden

		self.W_z = Parameter(torch.Tensor(self.n_in+self.n_hidden, self.n_hidden))
		self.W_r = Parameter(torch.Tensor(self.n_in+self.n_hidden, self.n_hidden))
		self.W = Parameter(torch.Tensor(self.n_in+self.n_hidden, self.n_hidden))

		self.weights = [self.W_z, self.W_r, self.W]

		# Initialization done according to paper
		self.init_parameters()

	def init_parameters(self):
		std = 1.0 / math.sqrt(self.n_hidden)
		for w in self.weights:
			w.data.uniform_(-std, std)

	def forward(self, x_t, h_t1):
		'''
		x_t : tensor of shape: (sequence length x batch size x input size)
		hidden: tensor of shape: (batch size x hidden size)
		out: will return tensor of shape: (sequence length x batch size x hidden size 
		'''
		xt_ht1 = torch.cat((x_t, h_t1), 1)
		z_t = torch.sigmoid(xt_ht1.mm(self.W_z))
		r_t = torch.sigmoid(xt_ht1.mm(self.W_r))

		xr_ht1 = torch.cat((torch.mul(r_t, h_t1), x_t), 1)
		h_hat_t = torch.tanh(xr_ht1.mm(self.W))

		h_t = torch.mul(1-z_t, h_t1) + torch.mul(z_t, h_hat_t)

		return h_t

class GRU(nn.Module):
	def __init__(self, n_in, n_hidden, batch_size, use_gpu):
		super(GRU, self).__init__()
		'''
		n_in: Number of Input Units
		n_hidden: Number of Hidden Units
		batch_size: Size of batch data
		use_gpu: Flag to use gpu 

		'''
		self.n_in = n_in
		self.n_hidden = n_hidden
		self.batch_size = batch_size
		self.lstm = GRUCell(self.n_in, self.n_hidden)
		self.use_gpu = use_gpu

	def init_hidden(self):
		if self.use_gpu:
			h0 = torch.zeros(self.batch_size, self.n_hidden).cuda()
		else:
			h0 = torch.zeros(self.batch_size, self.n_hidden)
		return h0

	def forward(self, x, batch_first=True):
		if batch_first:
			x = x.transpose(0, 1)
		seq_dim = x.shape[0]
		output = []
		self.hidden = self.init_hidden()
		for x_t in range(seq_dim):
			self.hidden = self.lstm(x[x_t], self.hidden)
			output.append(self.hidden)
		output = torch.stack(output, 0)
		return output