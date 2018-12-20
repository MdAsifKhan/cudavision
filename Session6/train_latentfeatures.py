import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from model import Encoder, Decoder
import torch
import torchvision.datasets as dsets
from tqdm import tqdm

def load_model(model_name, key='state_dict_encoder'):
    model_dir = '../Session6/model/' + model_name
    checkpoint = torch.load(model_dir)
    checkpoint = checkpoint[key]
    return checkpoint

def get_latent_features_cifar10(encoder, batch_size=100):
	trainset = dsets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
	testset = dsets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

	n_in = 32768  # 512x8x8
	train_features_ = np.zeros([len(trainset), n_in]) # Number of Samples by features
	train_labels_ = np.zeros(len(trainset))
	for b_idx, (train_data, train_labels) in tqdm(enumerate(trainloader)):
		latent_repr = encoder.forward(train_data)
		train_features_[b_idx: b_idx+batch_size,:] = latent_repr.detach().cpu().numpy()
		train_labels_[b_idx: b_idx+batch_size] = train_labels.detach().cpu().numpy()

	test_features_ = np.zeros([len(testset), n_in])
	test_features_ = np.zeros(len(testset))
	for b_idx, (test_data, test_labels) in enumerate(testloader):
		latent_repr = encoder.forward(train_data)
		test_features_[b_idx: b_idx+batch_size,:] = latent_repr.detach().cpu().numpy()
		test_labels_[b_idx: b_idx+batch_size] = test_labels.detach().cpu().numpy()

	return train_features_, train_labels_, test_features_, test_labels_

if __name__=='__main__':

	batch_size = 100
	lr = 0.001

	optim = 'adam'
	model_epoch = 5
	cudnn.benchmark = True
	add_noise = False
	if add_noise:
		model_name = 'AutoEncoder_lr_{}_opt_{}_epoch_{}_dae'.format(lr, optim, model_epoch)

	model_name = 'AutoEncoder_lr_{}_opt_{}_epoch_{}'.format(lr, optim, model_epoch)

	encoder = Encoder(batch_size=batch_size)
	decoder = Decoder(batch_size=batch_size)

	encoder.load_state_dict(load_model(model_name, key='state_dict_encoder'))
	decoder.load_state_dict(load_model(model_name, key='state_dict_decoder'))


	train_features_, train_labels_, test_features_, test_labels_ = get_latent_features_cifar10(encoder)
	predict_labels = OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_features_, train_labels_).predict(test_features_)
	print(classification_report(test_labels_, predict_labels))

	model =  LinearSVC(random_state=0)
	model = model.fit(train_features_soc, train_labels_soc)
	predict_labels_soc = model.predict(test_features_soc)
	print(classification_report(train_labels_soc, predict_labels_soc))