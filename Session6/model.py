import torch
import torch.nn as nn
import pdb

class Encoder(nn.Module):
    def __init__(self, batch_size):
        super(Encoder,self).__init__()
        self.batch_size = batch_size
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.Conv2d(128, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(256),
                        nn.Conv2d(256, 512, 3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(512),
                        nn.MaxPool2d(2),
                        nn.Conv2d(512, 512, 3, padding=1),
                        nn.ReLU()
        )
                
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(self.batch_size, -1)
        return out
    



class Decoder(nn.Module):
    def __init__(self, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.layer1 = nn.Sequential(
                        nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(512),
                        nn.ConvTranspose2d(512, 256, 3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(256),
                        nn.ConvTranspose2d(256, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(128)
        )
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.ConvTranspose2d(64, 32, 3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
                        nn.ReLU()
        )
        
    def forward(self, x):
        out = x.view(self.batch_size, 512, 8, 8)
        out = self.layer1(out)
        out = self.layer2(out)
        return out