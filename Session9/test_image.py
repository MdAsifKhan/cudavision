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
from utils import peak_detection, load_model, within_radius
import matplotlib.cm as cm
import math
import cv2
def prob_map(prob_map_, xmin, ymin, xmax, ymax, center, radius):
	'''
	get probability map based on center and radius
	'''
	for x in range(int(ymin), min(math.ceil(ymax), prob_map_.shape[0])):
		for y in range(int(xmin), min(math.ceil(xmax), prob_map_.shape[1])):
			prob_map_[x, y] = multivariate_normal.pdf([x, y], center, [radius, radius])
	return prob_map_

def get_center(output_nn, threshold, radius):
	'''
	Get center of ball using peak detection algorithm
	'''
	peak_ = peak_detection(threshold, output_nn)
	xmin, ymin = peak_[0][1] - radius, peak_[0][0] - radius
	xmax, ymax = peak_[0][1] + radius, peak_[0][0] + radius
	center = [(ymax+ymin)/2, (xmax+xmin)/2]
	return center, xmin, ymin, xmax, ymax 


def test_image(path, xml_path=None):
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
				prob_map_ = prob_map(prob_map_, xmin, ymin, xmax, ymax, center, radius)

		plt.imshow(prob_map_,  cmap=cm.jet)
		plt.savefig('{}/test_image_original_{}_drop_{}.png'.format(opt.result_root, opt.net, opt.drop_p))


	plt.cla()
	plt.clf()
	'''
	predicted map
	'''
	img = Image.open(path)
	transform = transforms.Compose([
			transforms.ColorJitter(brightness=0.3,
							contrast=0.4, saturation=0.4),		
			transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
	img = transform(img)
	img = img.numpy()
	img = img[np.newaxis, ...]
	img = torch.Tensor(img)
	nc = 1
	if opt.net=='net1':
		model = SweatyNet1(nc, opt.drop_p)
		print('SweatyNet1')
	elif opt.net=='net2':
		model = SweatyNet2(nc, opt.drop_p)
		print('SweatyNet2')
	elif opt.net=='net3':
		model = SweatyNet3(nc, opt.drop_p)
		print('SweatyNet3')
	else:
		raise ValueError('Model not supported')
	if opt.use_gpu:
		model = model.cuda()
		img = img.cuda()

	model_name = 'Model_lr_{}_opt_{}_epoch_{}_net_{}_drop_{}'.format(opt.lr, opt.optimizer, opt.test_epoch, opt.net, opt.drop_p)
	checkpoint, min_radius, threshold = load_model(model_name)
	model.load_state_dict(checkpoint)
	model.eval()

	with torch.no_grad():
		prob_map_predicted = np.zeros([120, 160], dtype='float32')
		output = model(img).cpu()
		output = output.squeeze().detach().numpy()
		plt.imshow(output,  cmap=cm.jet)
		plt.savefig('{}/test_image_predicted_{}_drop_{}.png'.format(opt.result_root, opt.net, opt.drop_p))
		plt.cla()
		plt.clf()

		binary_map = (output>0.1).astype(np.uint8)
		contours = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
		img = np.zeros(output.shape, np.uint8)
		if len(contours)>0:
			contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
			area, biggest_contour = max(contour_sizes, key=lambda x: x[0])
			cv2.drawContours(img, [biggest_contour], -1, (1), cv2.FILLED)
			processed_map = output * img
			predicted_center = peak_detection(processed_map, threshold)
		else:
			processed_map = output * img
			area = 0
			predicted_center = (-1, -1)

		plt.imshow(processed_map,  cmap=cm.jet)
		plt.savefig('{}/test_image_predicted_postprocess_{}_drop_{}.png'.format(opt.result_root, opt.net, opt.drop_p))
		plt.cla()
		plt.clf()


if __name__=='__main__':
	test_image(opt.image, opt.xml)