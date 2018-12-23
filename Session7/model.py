import torch
import torch.nn as nn
from LSTM import LSTM
from GRU import GRU

class LSTMModel(nn.Module):
	def __init__(self, n_in, n_hidden, n_out, batch_size, use_gpu):
		super(LSTMModel, self).__init__()
		self.n_in = n_in
		self.n_hidden = n_hidden
		self.n_out = n_out
		self.batch_size = batch_size
		self.use_gpu = use_gpu

		self.lstm_layer = LSTM(self.n_in, self.n_hidden, batch_size, self.use_gpu)
		self.clf = nn.Linear(self.n_hidden, self.n_out)
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x):
		output = self.lstm_layer(x)
		return self.clf(output[-1,:, :])


class GRUModel(nn.Module):
	def __init__(self, n_in, n_hidden, n_out, batch_size, use_gpu):
		super(GRUModel, self).__init__()
		self.n_in = n_in
		self.n_hidden = n_hidden
		self.n_out = n_out
		self.batch_size = batch_size
		self.use_gpu = use_gpu

		self.gru_layer = GRU(self.n_in, self.n_hidden, batch_size, self.use_gpu)
		self.clf = nn.Linear(self.n_hidden, self.n_out)
		self.loss = nn.CrossEntropyLoss()

	def forward(self, x):
		output = self.gru_layer(x)
		return self.clf(output[-1,:, :])