from model import SweatyNet1, ConvLSTM
from evaluator import ModelEvaluator
from arguments import opt
from dataset import SeqSoccerDataSet




trainset = SeqSoccerDataSet(data_path=opt.data_root + '/seq_train_cnn', map_file= 'train_maps', 
                                    transform= transforms.Compose([
                                    #transforms.RandomResizedCrop(opt.input_size[1]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(opt.rot_degree),
                                    transforms.ColorJitter(brightness=0.4,
                                                    contrast=0.4, saturation=0.4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))

testset = SeqSoccerDataSet(data_path=opt.data_root +'/seq_test_cnn', map_file='test_maps',
                                    transform= transforms.Compose([
                                    #transforms.RandomResizedCrop(opt.input_size[1]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(opt.rot_degree),
                                    transforms.ColorJitter(brightness=0.4,
                                                    contrast=0.4, saturation=0.4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]))




trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)


nc = 1
model = SweatyNet1(nc)
if opt.use_gpu:
    model = model.cuda()


epoch = 20 #Checkpoint model to load
model_name = opt.model_root + 'Model_lr_{}_opt_{}_epoch_{}'.format(opt.lr, opt.optimizer, epoch)
model.load_state_dict(load_model(model_name, key='state_dict_model'))

nc = 10 #Number of Sequence
map_size = [120, 160]
model.layer18 = ConvLSTM(nc, map_size)

opt.lr = 0.00001

modeleval = ModelEvaluator(model)
modeleval.evaluator(seq_trainloader, seq_testloader)
modeleval.plot_loss()



