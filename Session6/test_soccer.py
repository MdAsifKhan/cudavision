import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import numpy as np
from soccer_dataset import SoccerDataset

from model import Encoder, Decoder, LogisticRegression
from evaluator import ModelEvaluator
cudnn.benchmark = True


def test_img(path, encoder, model, model_epoch, encoder_epoch):
    img = Image.open(path)
    plt.imshow(img)
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img = transform(img)
    img = img.numpy()
    img = img[np.newaxis, ...]
    img = torch.Tensor(img)


    model.eval()
    encoder.eval()

    with torch.no_grad():
        latent_repr = encoder.forward(img)
        output = model.forward(latent_repr)
        _, idx = torch.max(test_preds.data, 1)
        print(str(output))
        print('prediction: %s' % ['not soccer', 'soccer'][idx])

def train_soccer(epochs, model, modeleval, dataloader):
	transform = transforms.Compose([
		transforms.RandomResizedCrop(32),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	for epoch in range(epochs):
    	model.train()

    	modeleval.train(encoder, epoch, dataloader, noise=False, print_every=100)

def run_trainer(dataloader, encoder, model, batch_size, epochs, lr=0.01, l2=0.0, use_gpu=True, optim='adam'):

	modeleval = ModelEvaluator(model, epochs, lr, l2=l2, use_gpu=use_gpu, optim=optim)
	
	modeleval.train(encoder, epoch, dataloader, noise=False, print_every=100)
	train_soccer(epochs, model, modeleval, dataloader)


if __name__ == '__main__':


	batch_size = 50

	encoder = Encoder(batch_size=batch_size)
	encoder_epoch = 10
	encoder.load_state_dict(load_encoder(epoch=encoder_epoch))

	n_in = 32768  # 512x8x8
	n_hidden = 512
	n_out = 10
	epochs = 20
	lr = 0.01

	model = LogisticRegression(n_in, n_hidden, n_out)
	dataset = SoccerDataset(transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3)
	run_trainer(dataloader, encoder, model, batch_size, epochs, lr)


	#model = LogisticRegression(n_in, n_hidden, n_out)
	#model_epoch = 10
	#model.load_state_dict(load_model(epoch=model_epoch))

	test_path = 'Session6/SoccerData/test'
	img1_path = test_path + '/test1.jpg'

	test_img(img1_path, encoder, model)