import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn


from model import Encoder, Decoder, LogisticRegression
from evaluator import ModelEvaluator, AutoEncoderEvaluator


cudnn.benchmark = True


trainset = dsets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
testset = dsets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# Set Hyperparameters
epochs = 10
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

modelname = 'model_learning_rate_{}_optimizer_{}_'.format(lr, opt)
torch.save([modeleval.encoder.state_dict(), modeleval.decoder.state_dict()], modelname)

#
n_in = 32768  # 512x8x8
n_hidden = 512
n_out = 10
model_epochs = 20
model_lr = 0.001
l2 = 0.0
#train classifier
encoder = AEeval.encoder
model = LogisticRegression(n_in, n_hidden, n_out)
modeleval = ModelEvaluator(model, model_epochs, model_lr, l2=l2, use_gpu=True, optim='adam')
acc_ = modeleval.evaluator(encoder, trainloader, testloader, noise=add_noise, print_every=100, validation=False)

print('Accuracy on test set {.2f}'.format(acc_))