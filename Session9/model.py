import torch
import torch.nn as nn
import pdb
import  torch.nn.functional as F

class SweatyNet1(nn.Module):
    def __init__(self, nc):
        super(SweatyNet1,self).__init__()
        self.nc = nc
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 8, 3, padding=1),
                        nn.BatchNorm2d(8),
                        nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, stride=2)
        
        self.layer2 = nn.Sequential(
                        nn.Conv2d(8, 16, 3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU()

        )
        self.layer3 = nn.Sequential(
                        nn.Conv2d(16, 16, 3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU()

        )
        self.layer4 = nn.Sequential(
                        nn.Conv2d(8+16, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU()

        )
        self.layer5 = nn.Sequential(
                        nn.Conv2d(32, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU()

        )
        self.layer6 = nn.Sequential(
                        nn.Conv2d(8+16+32, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )                
        self.layer7 = nn.Sequential(
                        nn.Conv2d(64, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )
        self.layer8 = nn.Sequential(
                        nn.Conv2d(64, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )
        self.layer9 = nn.Sequential(
                        nn.Conv2d(8+16+32+64, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU()

        )
        self.layer10 = nn.Sequential(
                        nn.Conv2d(128, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU()

        )
        self.layer11 = nn.Sequential(
                        nn.Conv2d(128, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU()

        )
        self.layer12 = nn.Sequential(
                        nn.Conv2d(128, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.layer13 = nn.Sequential(
                        nn.Conv2d(8+16+32+64+64, 64, 1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )  
        self.layer14 = nn.Sequential(
                        nn.Conv2d(64, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU()

        )
        self.layer15 = nn.Sequential(
                        nn.Conv2d(32, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU()

        )
        self.layer16 = nn.Sequential(
                        nn.Conv2d(8+16+32+32, 16, 1, padding=0),
                        nn.BatchNorm2d(16),
                        nn.ReLU()

        )       
        self.layer17 = nn.Sequential(
                        nn.Conv2d(16, 16, 3, padding=0),
                        nn.BatchNorm2d(16),
                        nn.ReLU()

        )
        self.layer18 = nn.Sequential(
                        nn.Conv2d(16, self.nc, 3, padding=0),
                        nn.BatchNorm2d(self.nc),
                        nn.ReLU()

        ) 
    def forward(self,x):
        out1 = self.pool(self.layer1(x))

        out2 = torch.cat((out1, self.layer3(self.layer2(out1))), 1)
        out2 = self.pool(out2)
        out3 = torch.cat((out2, self.layer5(self.layer4(out2))), 1)

        out4 = self.pool(out3)
        out5 = torch.cat((out4, self.layer8(self.layer7(self.layer6(out4)))), 1)

        out6 = self.pool(out5)
        out7 = self.layer12(self.layer11(self.layer10(self.layer9(out6))))

        out7 = torch.cat((out5, self.upsample(out7)), 1)

        out8 = self.upsample(self.layer15(self.layer14(self.layer13(out7))))
        out3 = F.pad(out3, pad=(2,2,2,2), mode='constant', value=0)
        out8 = torch.cat((out8, out3), 1)

        out = self.layer18(self.layer17(self.layer16(out8)))

        return out.squeeze()

class SweatyNet2(nn.Module):
    def __init__(self, nc):
        super(SweatyNet2,self).__init__()
        self.nc = nc
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 8, 3, padding=0),
                        nn.BatchNorm2d(8),
                        nn.ReLU()
        )
        self.pool = nn.nn.MaxPool2d(2)
        
        self.layer2 = nn.Sequential(
                        nn.Conv2d(8, 16, 3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU()

        )
        self.layer3 = nn.Sequential(
                        nn.Conv2d(8+16, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU()

        )
        self.layer4 = nn.Sequential(
                        nn.Conv2d(8+16+32, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )                
        self.layer5 = nn.Sequential(
                        nn.Conv2d(64, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )

        self.layer6 = nn.Sequential(
                        nn.Conv2d(8+16+32+64, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU()

        )
        self.layer7 = nn.Sequential(
                        nn.Conv2d(128, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU()

        )
        self.layer8 = nn.Sequential(
                        nn.Conv2d(128, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.layer9 = nn.Sequential(
                        nn.Conv2d(8+16+32+64+64, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )
        self.layer10 = nn.Sequential(
                        nn.Conv2d(64, 32, 1, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU()

        )  
        self.layer11 = nn.Sequential(
                        nn.Conv2d(64, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU()

        )
        self.layer12 = nn.Sequential(
                        nn.Conv2d(8+16+32+32, 16, 1, padding=0),
                        nn.BatchNorm2d(16),
                        nn.ReLU()

        )       
        self.layer13 = nn.Sequential(
                        nn.Conv2d(16, 16, 3, padding=0),
                        nn.BatchNorm2d(16),
                        nn.ReLU()

        )
        self.layer14 = nn.Sequential(
                        nn.Conv2d(16, self.nc, 3, padding=0),
                        nn.BatchNorm2d(self.nc),
                        nn.ReLU()

        ) 

    def forward(self,x):
        out1 = self.pool(self.layer1(x))
        
        out2 = torch.cat((out1, self.layer2(out1)), 1)
        out2 = self.pool(out2)
        out3 = torch.cat((out2, self.layer3(out2)), 1)

        out4 = self.pool(out3)
        out5 = torch.cat((out4, self.layer5(self.layer4(out4))), 1)

        out6 = self.pool(out5)
        out7 = self.layer8(self.layer7(self.layer6(out6)))
        out7 = torch.cat((out5, self.upsample(out7)), 1)

        out8 = self.upsample(self.layer11(self.layer10(self.layer9(out7))))

        out3 = F.pad(out3, pad=(2,2,2,2), mode='constant', value=0)
        out8 = torch.cat((out8, out3), 1)

        out = self.layer14(self.layer13(self.layer12(out8)))
        
        return out.squeeze()

class SweatyNet3(nn.Module):
    def __init__(self, nc):
        super(SweatyNet3, self).__init__()
        self.nc = nc
        
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3, 8, 3, padding=0),
                        nn.BatchNorm2d(8),
                        nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2)
        
        self.layer2 = nn.Sequential(
                        nn.Conv2d(8, 8, 1, padding=1),
                        nn.BatchNorm2d(8),
                        nn.ReLU()
        )
        self.layer3 = nn.Sequential(
                        nn.Conv2d(8, 16, 3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU()

        )
        self.layer4 = nn.Sequential(
                        nn.Conv2d(8+16, 16, 1, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU()

        )
        self.layer5 = nn.Sequential(
                        nn.Conv2d(16, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU()

        )
        self.layer6 = nn.Sequential(
                        nn.Conv2d(8+16+32, 32, 1, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU()

        )                
        self.layer7 = nn.Sequential(
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )
        self.layer8 = nn.Sequential(
                        nn.Conv2d(64, 32, 1, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU()

        )
        self.layer9 = nn.Sequential(
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )
        self.layer10 = nn.Sequential(
                        nn.Conv2d(8+16+32+64, 64, 1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )
        self.layer11 = nn.Sequential(
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU()

        )
        self.layer12 = nn.Sequential(
                        nn.Conv2d(128, 64, 1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )
        self.layer13 = nn.Sequential(
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU()
        )
        self.layer14 = nn.Sequential(
                        nn.Conv2d(128, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.layer15 = nn.Sequential(
                        nn.Conv2d(8+16+32+64+64, 64, 1, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()

        )  
        self.layer16 = nn.Sequential(
                        nn.Conv2d(64, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU()

        )
        self.layer17 = nn.Sequential(
                        nn.Conv2d(32, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU()

        )
        self.layer18 = nn.Sequential(
                        nn.Conv2d(8+16+32+32, 16, 1, padding=0),
                        nn.BatchNorm2d(16),
                        nn.ReLU()

        )       
        self.layer19 = nn.Sequential(
                        nn.Conv2d(16, 16, 3, padding=0),
                        nn.BatchNorm2d(16),
                        nn.ReLU()

        )
        self.layer20 = nn.Sequential(
                        nn.Conv2d(16, self.nc, 3, padding=0),
                        nn.BatchNorm2d(self.nc),
                        nn.ReLU()

        ) 
    def forward(self, x):
        out1 = self.pool(self.layer1(x))
        out2 = torch.cat((out1, self.layer3(self.layer2(out1))), 1)
        out2 = self.pool(out2)

        out3 = torch.cat((out2, self.layer5(self.layer4(out2))), 1)

        out4 = self.pool(out3)
        out5 = torch.cat((out4, self.layer9(self.layer8(self.layer7(self.layer6(out4))))), 1)

        out6 = self.pool(out5)
        out7 = self.layer14(self.layer13(self.layer12(self.layer11(self.layer10(out6)))))
        out7 = torch.cat((out5, self.upsample(out7)), 1)

        out8 = self.upsample(self.layer17(self.layer16(self.layer15(out7))))

        out3 = F.pad(out3, pad=(2,2,2,2), mode='constant', value=0)
        out8 = torch.cat((out8, out3), 1)

        out = self.layer20(self.layer19(self.layer18(out8)))
        
        return out.squeeze()