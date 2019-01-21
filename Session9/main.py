import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from model import SweatyNet1, SweatyNet2, SweatyNet3
from evaluator import ModelEvaluator
from arguments import opt
from dataset import SoccerDataSet
import random


random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True


# Set Hyperparameters
epochs = opt.nm_epochs
batch_size = opt.batch_size

#
if opt.dataset== 'soccer':
	trainset = SoccerDataSet(data_path=opt.data_root + '/train_cnn', map_file= 'train_maps', 
										transform= transforms.Compose([
										#transforms.RandomResizedCrop(opt.input_size[1]),	
										#transforms.RandomHorizontalFlip(),	
										#transforms.RandomRotation(opt.rot_degree),
										transforms.ColorJitter(brightness=0.3,
														contrast=0.4, saturation=0.4),
										transforms.ToTensor(),
										transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
				]))

	testset = SoccerDataSet(data_path=opt.data_root +'/test_cnn', map_file='test_maps',
										transform= transforms.Compose([
										#transforms.RandomResizedCrop(opt.input_size[1]),	
										#transforms.RandomHorizontalFlip(),	
										#transforms.RandomRotation(opt.rot_degree),
										transforms.ColorJitter(brightness=0.3,
														contrast=0.4, saturation=0.4),
										transforms.ToTensor(),
										transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
				]))
else:
	pass

# Data Loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)
import pdb
# Pytorch Cross Entropy Loss
nc = 1
threshold = trainset.threshold
model = SweatyNet1(nc)
modeleval = ModelEvaluator(model, threshold)
modeleval.evaluator(trainloader, testloader)
modeleval.plot_loss()