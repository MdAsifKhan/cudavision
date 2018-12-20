import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import numpy as np
from soccer_dataset import SoccerDataset

from model import Encoder, Decoder
cudnn.benchmark = True

def load_model(model_name, key='state_dict_encoder'):
    model_dir = '../Session6/model/' + model_name
    checkpoint = torch.load(model_dir)
    checkpoint = checkpoint[key]
    return checkpoint

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


def get_features_soccer(encoder, batch_size=100):
	transform = transforms.Compose([
					transforms.Resize(64),
					transforms.CenterCrop(32),
					transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

	dataset = SoccerDataset(transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=3)
	n_in = 32768  # 512x8x8
	train_features_ = np.zeros([len(dataset), n_in]) # Number of Samples by features
	train_labels_ = np.zeros(len(dataset))
	for b_idx, (train_data, train_labels) in tqdm(enumerate(dataloader)):
		latent_repr = encoder.forward(train_data)
		train_features_[b_idx: b_idx+batch_size,:] = latent_repr.detach().cpu().numpy()
		train_labels_[b_idx: b_idx+batch_size] = train_labels.detach().cpu().numpy()

	return train_features_, train_labels_

if __name__ == '__main__':


	batch_size = 100
	lr = 0.001

	optim = 'adam'
	model_epoch = 5
	use_gpu= True
	cudnn.benchmark = True
	add_noise = False
	if add_noise:
		model_name = 'AutoEncoder_lr_{}_opt_{}_epoch_{}_dae'.format(lr, optim, model_epoch)
	model_name = 'AutoEncoder_lr_{}_opt_{}_epoch_{}'.format(lr, optim, model_epoch)

	encoder = Encoder(batch_size=batch_size)
	decoder = Decoder(batch_size=batch_size)

	encoder.load_state_dict(load_model(model_name, key='state_dict_encoder'))
	decoder.load_state_dict(load_model(model_name, key='state_dict_decoder'))

	n_in = 32768  # 512x8x8
	

	train_features_, train_labels_ = get_features_soccer(encoder)


	test_path = 'Session6/SoccerData/test'
	img1_path = test_path + '/test1.jpg'

	test_img(img1_path, encoder, model)