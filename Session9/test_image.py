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

def load_model(model_name, key='state_dict_model'):
	model_dir = opt.model_root + model_name
	checkpoint = torch.load(model_dir)
	checkpoint = checkpoint[key]
	return checkpoint

def prob_map(prob_map_, center, radius):
	for x in range(prob_map_.shape[0]):
		for y in range(prob_map_.shape[1]):
			prob_map_[x, y] = multivariate_normal.pdf([x, y], center, [2*radius, 2*radius])
	return prob_map_

def test_image(path, xml_path=None, epoch=15):
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
	model.load_state_dict(load_model(model_name, key='state_dict_model'))
	model.eval()

	with torch.no_grad():
		output = model(img).transpose(1, 2).cpu().numpy()
		plt.imshow(output,  cmap='hot', interpolation='nearest')
		plt.savefig('{}/test_image_predicted.png'.format(opt.result_root))

	plt.cla()
	plt.clf()

	if xml_path is not None:
		with open(xml_path) as f:
			tree = xd.parse(f.read())

		if type(tree['annotation']['object']) is not list:
			tree['annotation']['object'] = [tree['annotation']['object']]

		prob_map_ = np.zeros([160, 120], dtype='float32')
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

if __name__ =='__main__':
	img_path = opt.data_root + '/test_cnn/'+ '00292.jpg'
	xml_path = opt.data_root + '/test_cnn/'+ '00292.xml'
	epoch = 30
	test_image(img_path, xml_path)
