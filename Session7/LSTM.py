import torch
import torch.nn as nn
from torch.nn import Parameter
import math


class LSTMCell(nn.Module):
	def __init__(self, n_in, n_hidden):
		'''
		Hochreiter, Sepp, and Jürgen Schmidhuber. 
		'Long short-term memory.'' Neural computation 9.8 (1997): 1735-1780.
		'''

		super(LSTMCell, self).__init__()
		'''
		n_in: Number of Input Units
		n_hidden: Number of Hidden Units
		'''
		self.n_in = n_in
		self.n_hidden = n_hidden

		self.W_i = Parameter(torch.Tensor(self.n_in+self.n_hidden, self.n_hidden))
		self.W_o = Parameter(torch.Tensor(self.n_in+self.n_hidden, self.n_hidden))
		self.W_f = Parameter(torch.Tensor(self.n_in+self.n_hidden, self.n_hidden))

		self.W_c = Parameter(torch.Tensor(self.n_in+self.n_hidden, self.n_hidden))

		self.b_i = Parameter(torch.Tensor(self.n_hidden))
		self.b_o = Parameter(torch.Tensor(self.n_hidden))
		self.b_f = Parameter(torch.Tensor(self.n_hidden))
		self.b_c = Parameter(torch.Tensor(self.n_hidden))

		self.weights = [self.W_i, self.W_o, self.W_f, self.W_c]
		self.bias = [self.b_i, self.b_o, self.b_f, self.b_c]

		self.init_parameters()

	def init_parameters(self):
		std = 1.0 / math.sqrt(self.n_hidden)
		for w in self.weights:
			w.data.uniform_(-std, std)
		for b in self.bias:
			nn.init.constant_(b.data, val=0.)

	def forward(self, x_t, hidden):
		'''
		x_t : sequence length x batch size x input size
		hidden: batch size x hidden size
		out: sequence length x batch size x hidden size 
		'''
		h_t1, c_t1 = hidden

		xt_ht1 = torch.cat((x_t, h_t1), 1)
		i_t = torch.sigmoid(xt_ht1.mm(self.W_i) + self.b_i)
		f_t = torch.sigmoid(xt_ht1.mm(self.W_f) + self.b_f)

		c_hat_t = torch.tanh(xt_ht1.mm(self.W_c) + self.b_c)

		c_t = torch.mul(f_t, c_t1) + torch.mul(i_t, c_hat_t)

		o_t = torch.sigmoid(xt_ht1.mm(self.W_o) + self.b_o)

		h_t = torch.mul(o_t, torch.tanh(c_t))

		return c_t, h_t

class LSTM(nn.Module):
	def __init__(self, n_in, n_hidden, batch_size, use_gpu):
		super(LSTM, self).__init__()
		'''
		n_in: Number of Input Units
		n_hidden: Number of Hidden Units
		batch_size: Size of batch data
		use_gpu: Flag to use gpu 

		'''
		self.n_in = n_in
		self.n_hidden = n_hidden
		self.batch_size = batch_size
		self.lstm = LSTMCell(self.n_in, self.n_hidden)
		self.use_gpu = use_gpu

	def init_hidden(self):
		if self.use_gpu:
			h0 = torch.zeros(self.batch_size, self.n_hidden).cuda()
			c0 = torch.zeros(self.batch_size, self.n_hidden).cuda()
		else:
			h0 = torch.zeros(self.batch_size, self.n_hidden).cuda()
			c0 = torch.zeros(self.batch_size, self.n_hidden).cuda()
		return h0, c0

	def forward(self, x, batch_first=True):
		if batch_first:
			x = x.transpose(0, 1)
		seq_dim = x.shape[0]
		output = []
		self.hidden = self.init_hidden()
		for x_t in range(seq_dim):
			self.hidden = self.lstm(x[x_t], self.hidden)
			output.append(self.hidden[0])
		output = torch.stack(output, 0)
		return output