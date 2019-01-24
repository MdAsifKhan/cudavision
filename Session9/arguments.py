import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=False, help='soccer| sequence', default='soccer')
parser.add_argument('--net', required=False, help='net1| net2| net3', default='net1')
parser.add_argument('--data_root', required=False, help='path to dataset', default='')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')

parser.add_argument('--nm_epochs', type=int, default=50, help='number of epochs for training')
parser.add_argument('--l2', type=float, default=0.0, help='l2 parameter')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')
parser.add_argument('--print_every', type=int, default=2, help='print checkpoints')
parser.add_argument('--save_every', type=int, default=5, help='model checkpoints')

parser.add_argument('--drop_p', type=float, default=0.0, help='Dropout Probability')
parser.add_argument('--resume', type=int, default=0, help='epoch at which training resumes')
parser.add_argument('--model_root', default='', help='folder to output model checkpoints')
parser.add_argument('--result_root', default='', help='folder to output image checkpoints')
parser.add_argument('--use_gpu', type=bool, default=True, help='Enable GPU Training')
parser.add_argument('--manualSeed', type=int, default=123, help='manual seed')
parser.add_argument('--image', required=False, help='test image path', default='')
parser.add_argument('--xml', required=False, help='test xml path', default='')

opt = parser.parse_args()
opt.dataset = 'soccer'
opt.input_size = (640, 480)
opt.rot_degree = 45
if opt.data_root == '':
	opt.data_root = '/home/local/stud/khan01/cudavision/Session9/SoccerData'
if opt.model_root == '':
	opt.model_root = '/home/local/stud/khan01/cudavision/Session9/model/'
if opt.result_root == '':
	opt.result_root = '/home/local/stud/khan01/cudavision/Session9/results'
if opt.image == '':
	opt.image = opt.data_root + '/test_cnn/' + '00292.jpg'
if opt.xml == '':
	opt.xml = opt.data_root + '/test_cnn/' + '00292.xml'
if opt.optimizer=='adam':
	opt.lr = 0.001

elif opt.optimizer=='sgd':
	opt.lr = 0.001
	opt.mom = 0.9
elif opt.optimizer=='adadelta':
	opt.lr = 0.002
	opt.eps = 1e-6
	opt.decay = 0.0

elif opt.optimizer=='rmsprop':
	opt.lr = 0.002
	opt.eps = 1e-7
	opt.decay = 0.0
	opt.alpha = 0.995
else:
	ValueError('Optimizer Not Supported')
