from model import SweatyNet1
from evaluator import ModelEvaluator
from arguments import opt
from PIL import Image
import xmltodict as xd
from torchvision import transforms
import torch
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import pdb

def load_model(model_name, key='state_dict_model', thresh='threshold'):
	model_dir = opt.model_root + model_name
	checkpoint = torch.load(model_dir)
	checkpoint, threshold = checkpoint[key], checkpoint[thresh]
	return checkpoint, threshold

def prob_map(prob_map_, center, radius):
	for x in range(prob_map_.shape[0]):
		for y in range(prob_map_.shape[1]):
			prob_map_[x, y] = multivariate_normal.pdf([x, y], center, [2*radius, 2*radius])
	return prob_map_

def get_center(output_nn, threshold, radius):
	peak =  np.unravel_index(np.argmax(output_nn, axis=None), output_nn.shape)
	max_value = output_nn[peak]
	while max_value>threshold:
		map_[peak] = 0
		peak = np.unravel_index(np.argmax(output_nn, axis=None), output_nn.shape)
		max_value = map_[peak]
	xmin, ymin = peak[1] - radius, peak[0] - radius
	xmax, ymax = peak[1] + radius, peak[0] + radius
	center = [(ymax+ymin)/2, (xmax+xmin)/2]
	return center


def test_image(path, xml_path=None, epoch=15):
	'''
	Original Map
	'''
	if xml_path is not None:
		with open(xml_path) as f:
			tree = xd.parse(f.read())

		if type(tree['annotation']['object']) is not list:
			tree['annotation']['object'] = [tree['annotation']['object']]

		prob_map_ = np.zeros([120, 160], dtype='float32')
		for object_ in tree['annotation']['object']:
			if object_['name']=='ball':
				bndbox = object_['bndbox']
				xmin, ymin = int(bndbox['xmin'])/4, int(bndbox['ymin'])/4
				xmax, ymax = int(bndbox['xmax'])/4, int(bndbox['ymax'])/4
				center = [(ymax+ymin)/2, (xmax+xmin)/2]
				radius = min((xmax-xmin)/2, (ymax-ymin)/2)
				prob_map_ = prob_map(prob_map_, center, radius)

		plt.imshow(prob_map_,  cmap='hot', interpolation='nearest')
		plt.savefig('{}/test_image_original.png'.format(opt.result_root))


	plt.cla()
	plt.clf()
	'''
	predicted map
	'''
	img = Image.open(path)
	transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5),
					(0.5, 0.5, 0.5))])
	img = transform(img)
	img = img.numpy()
	img = img[np.newaxis, ...]
	img = torch.Tensor(img)

	model = SweatyNet1(nc=1)
	if opt.use_gpu:
		model = model.cuda()
		img = img.cuda()

	model_name = 'Model_lr_{}_opt_{}_epoch_{}'.format(opt.lr, opt.optimizer, epoch)
	checkpoint, threshold = load_model(model_name)
	model.load_state_dict(checkpoint)
	model.eval()

	with torch.no_grad():
		prob_map_predicted = np.zeros([120, 160], dtype='float32')
		output = model(img).cpu().numpy()
		center = get_center(output, threshold, radius)
		prob_map_predicted = prob_map(prob_map_predicted, center, radius)
		plt.imshow(prob_map_predicted,  cmap='hot', interpolation='nearest')
		plt.savefig('{}/test_image_predicted.png'.format(opt.result_root))



if __name__=='__main__':
	epoch = 20
	test_image(opt.image, opt.xml)

