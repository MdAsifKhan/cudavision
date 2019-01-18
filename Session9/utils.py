import numpy as np
import pdb

def batch_iou(a, b, epsilon=1e-5):
	""" 
	http://ronny.rest/tutorials/module/localization_001/iou/

	Given two arrays `a` and `b` where each row contains a bounding
	box defined as a list of four numbers:
		[x1,y1,x2,y2]
	where:
		x1,y1 represent the upper left corner
		x2,y2 represent the lower right corner
		It returns the Intersect of Union scores for each corresponding
		pair of boxes.

	Args:
		a:			(numpy array) each row containing [x1,y1,x2,y2] coordinates
		b:			(numpy array) each row containing [x1,y1,x2,y2] coordinates
		epsilon:	(float) Small value to prevent division by zero

	Returns:
		(numpy array) The Intersect of Union scores for each pair of bounding
		boxes.
	"""
	# COORDINATES OF THE INTERSECTION BOXES
	x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
	y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
	x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
	y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

	# AREAS OF OVERLAP - Area where the boxes intersect
	width = (x2 - x1)
	height = (y2 - y1)

	# handle case where there is NO overlap
	width[width < 0] = 0
	height[height < 0] = 0

	area_overlap = width * height

	# COMBINED AREAS
	area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
	area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
	area_combined = area_a + area_b - area_overlap

	# RATIO OF AREA OF OVERLAP OVER COMBINED AREA
	iou = area_overlap / (area_combined + epsilon)
	
	return iou

def predict_box(peaks_predicted, box_actual):
	'''
	get predicted box
	'''
	box_predicted = []
	for box, peak in zip(box_actual.tolist(), peaks_predicted):
	    radius = min((box[2]-box[0])/2, (box[3]-box[1])/2)
	    xmin, ymin = peak[1] - radius, peak[0] - radius
	    xmax, ymax = peak[1] + radius, peak[0] + radius
	    box_predicted.append(np.array([xmin, ymin, xmax, ymax]))

	return np.asarray(box_predicted)


def performance_metric(box_actual, box_predicted):
	'''
	Performance Metric based on predicted peaks
	'''

	iou = batch_iou(box_actual, box_predicted)
	TP = iou>0.5
	FP = iou<=0.5
	FN = iou<=1e-5

	accuracy = TP.sum()/len(TP)
	FDR = FP.sum()/(FP.sum() + TP.sum())
	RC = TP.sum()/(TP.sum() + FN.sum())

	return FDR, RC, accuracy

def peak_detection(threshold, maps):
	'''
	peak detection on predicted score
	'''
	peaks = []
	for map_ in maps:
		peak =  np.unravel_index(np.argmax(map_, axis=None), map_.shape)
		max_value = map_[peak]
		while max_value>threshold:
			map_[peak] = 0
			peak = np.unravel_index(np.argmax(map_, axis=None), map_.shape)
			max_value = map_[peak]
		peaks.append(peak)
	
	return  peaks

def load_model(model_name, key='state_dict_model', thresh='threshold'):
	'''
	get checkpoint of trained model and threshold
	'''
	model_dir = opt.model_root + model_name
	checkpoint = torch.load(model_dir)
	checkpoint, threshold = checkpoint[key], checkpoint[thresh]
	
	return checkpoint, threshold