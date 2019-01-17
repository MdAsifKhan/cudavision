from model import SweatyNet1
from evaluator import ModelEvaluator
from arguments import opt
from PIL import Image
from torchvision import transforms
import torch
from matplotlib import pyplot as plt
import numpy as np

def load_model(model_name, key='state_dict_model'):
        model_dir = opt.model_root + model_name
        checkpoint = torch.load(model_dir)
        checkpoint = checkpoint[key]
        return checkpoint

def test_image(path, epoch=15):
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
		output = model(img).squeeze().cpu().numpy()
		plt.imshow(output,  cmap='hot', interpolation='nearest')
		plt.savefig('{}/test_image.png'.format(opt.result_root))


if __name__ =='__main__':
	img_path = opt.data_root + '/test_cnn/'+ '00255.jpg'
	test_image(img_path)
