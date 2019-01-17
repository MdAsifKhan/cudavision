import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=False, help='soccer| sequence', default='soccer')
parser.add_argument('--data_root', required=False, help='path to dataset', default='')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')

parser.add_argument('--nm_epochs', type=int, default=25, help='number of epochs for training')
parser.add_argument('--l2 regularization', type=float, default=1e-5, help='l2 parameter')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')
parser.add_argument('--print_every', type=int, default=5, help='print checkpoints')
parser.add_argument('--save_every', type=int, default=5, help='model checkpoints')

parser.add_argument('--resume', type=int, default=0, help='epoch at which training resumes')
parser.add_argument('--model_root', default='', help='folder to output model checkpoints')
parser.add_argument('--result_root', default='', help='folder to output image checkpoints')
parser.add_argument('--use_gpu', type=bool, default=True, help='Enable GPU Training')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
opt.dataset = 'soccer'
opt.manualSeed = 123
opt.input_size = (640, 480)
opt.rot_degree = 45
if opt.data_root == '':
	opt.data_root = '/home/local/stud/khan01/cudavision/Session9/SoccerData'
if opt.model_root == '':
	opt.model_root = '/home/local/stud/khan01/cudavision/Session9/model/'
if opt.result_root == '':
	opt.result_root = '/home/local/stud/khan01/cudavision/Session9/results'


if opt.optimizer=='adam':
	opt.lr = 0.0001

elif opt.optimizer=='sgd':
	opt.lr = 0.0001
	opt.mom = 0.9
elif opt.optimizer=='adadelta':
	opt.lr = 0.0002
	opt.eps = 1e-6
	opt.decay = 0.0

elif opt.optimizer=='rmsprop':
	opt.lr = 0.0002
	opt.eps = 1e-7
	opt.decay = 0.0
	opt.alpha = 0.995
else:
	ValueError('Optimizer Not Supported')

opt.l2 = 0.0
opt.nm_epochs = 50
opt.save_every = 5
opt.batch_size = 8
opt.print_every = 2