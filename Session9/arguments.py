import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=False, help='soccer| sequence', default='soccer')
parser.add_argument('--data_root', required=False, help='path to dataset', default='')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=6, help='input batch size')

parser.add_argument('--nm_epochs', type=int, default=100, help='number of epochs for training')
parser.add_argument('--l2', type=float, default=0.0, help='l2 parameter')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer to use')
parser.add_argument('--print_every', type=int, default=2, help='print checkpoints')
parser.add_argument('--save_every', type=int, default=5, help='model checkpoints')
parser.add_argument('--weight_decay', default=1e-4, help='regularization constant for l_2 regularizer of W')

parser.add_argument('--resume', type=int, default=0, help='epoch at which training resumes')
parser.add_argument('--model_root', default='', help='folder to output model checkpoints')
parser.add_argument('--result_root', default='', help='folder to output image checkpoints')
parser.add_argument('--use_gpu', type=bool, default=True, help='Enable GPU Training')
parser.add_argument('--manualSeed', type=int, default=123, help='manual seed')
parser.add_argument('--image', required=False, help='test image path', default='')
parser.add_argument('--xml', required=False, help='test xml path', default='')


######################################################################################
# sequential part

parser.add_argument('--seq_model', default='tcn', help='lstm | tcn')

parser.add_argument('--seq_dataset', default='toy.seq/npy')
parser.add_argument('--seq_dataset_root', default='')
parser.add_argument('--seq_save_out', default='seq_output')

###########################################
# tcn
parser.add_argument('--hist', default=20)
parser.add_argument('--nhid', default=25, type=int)
parser.add_argument('--output_size', default=2, type=int)
parser.add_argument('--levels', default=3, type=int)
parser.add_argument('--ksize', default=7, type=int)
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')

###########################################
# balls
parser.add_argument('--map_size_x', default=120, type=int)
parser.add_argument('--map_size_y', default=160, type=int)
parser.add_argument('--window_size', default=15, type=int)
parser.add_argument('--n_balls', default=1, type=int)
parser.add_argument('--min_sigma', default=2, type=int)
parser.add_argument('--max_sigma', default=5, type=int)
parser.add_argument('--max_shift', default=300, type=int)
parser.add_argument('--max_move_steps', default=60, type=int)
parser.add_argument('--min_move_steps', default=30, type=int)

###########################################
# save
parser.add_argument('--seq_resume', default=True, type=bool,
                    help='load model for embeddings, if positive then it is number of '
                         'epoch which should be loaded')
parser.add_argument('--seq_resume_str',
                    default='model/tcn.ep100_95.pth.tar')
                    # default='model/lstm.ep30_20.pth.tar')
parser.add_argument('--sweaty_resume_str', default='model/Model_lr_0.001_opt_adam_epoch_25')
parser.add_argument('--seq_save_model', default='ft.tcn.sw.seq.')

opt = parser.parse_args()
opt.dataset = 'soccer'
opt.input_size = (640, 480)
opt.rot_degree = 45
if opt.data_root == '':
    opt.data_root = 'SoccerData'
if opt.model_root == '':
    opt.model_root = 'model/'
if opt.result_root == '':
    opt.result_root = 'results'
if opt.image == '':
    opt.image = opt.data_root + '/test_cnn/' + '00292.jpg'
if opt.xml == '':
    opt.xml = opt.data_root + '/test_cnn/' + '00292.xml'
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
