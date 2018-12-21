import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


from model import Encoder, Decoder
from evaluator import AutoEncoderEvaluator


def load_encoder(encoder_name):
	model_dir = '../Session6/model/' + encoder_name
	checkpoint = torch.load(model_dir)
	checkpoint = checkpoint['state_dict_encoder']
	return checkpoint

def load_model(model_name):
	model_dir = '../Session6/model/' + model_name
	checkpoint = torch.load(model_dir)
	checkpoint = checkpoint['state_dict']
	return checkpoint

cudnn.benchmark = True


trainset = dsets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
testset = dsets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# Set Hyperparameters
epochs = 20
batch_size = 50
lr = 0.001

# Data Loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)


# Model
l2 = 0.0
optim = 'adam'
# Pytorch Cross Entropy Loss
encoder = Encoder(batch_size=batch_size)
decoder = Decoder(batch_size=batch_size)
add_noise = False
AEeval = AutoEncoderEvaluator(encoder, decoder, epochs, lr, batch_size=batch_size, l2=l2, add_noise=add_noise, use_gpu=True, optim=optim)
AEeval.evaluator(trainloader, testloader, print_every=100)