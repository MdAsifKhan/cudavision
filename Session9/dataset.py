from arguments import opt
import os
import pdb
import xmltodict as xd
import numpy as np
from torchvision import transforms
from scipy.stats import multivariate_normal
from PIL import Image
import torch

class SoccerDataSet:
	'''
	DataSet reader: Readet to get images and probability map from a folder
	'''
	def __init__(self, data_path, map_file, transform=None):
		self.dataroot = data_path
		self.map_file = map_file
		self.transform = transform
		with h5py.File(self.dataroot + '/' + self.map_file,'r') as hf:
			 self.targets = hf.get('prob_maps').tolist()
			 self.filenames = hf.get('filenames').tolist()
			 self.box = hf.get('ros').tolist()

		self.images, self.targets = [], []
		for filename in self.filenames:
			filename = filename + '.jpg'
			self.images.append(os.path.join(self.dataroot, filename))

	def __len__(self):
		return len(self.targets)

	def __getitem__(self, idx):
		img_name = self.images[idx]
		img_path = os.path.join(self.dataroot, img_name)
		
		img = Image.open(img_path)
		
		if self.transform:
			img = self.transform(img)

		name = img_name[:-4]
		idt = self.filenames.index(name)
		prob_ = self.targets[idt]
		coord_ = self.box[idt]
		return img, prob_, coord_