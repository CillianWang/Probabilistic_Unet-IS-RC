from calendar import c
import torch
import numpy as np
from models.usleep_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, classifier=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = EnConv(n_channels, 5)
        self.down1 = EnConv(5, 7)
        self.down2 = EnConv(7, 10)
        self.down3 = EnConv(10, 14)
        self.down4 = EnConv(14, 20)
        self.down5 = EnConv(20, 28)
        self.down6 = EnConv(28, 40)
        self.down7 = EnConv(40, 57)
        self.down8 = EnConv(57, 81)
        self.down9 = EnConv(81, 115)
        self.down10 = EnConv(115, 163)
        self.down11 = EnConv(163, 231)
        self.down12 = EnConv(231, 327)
        # above are 12 layers' decoder, with sqrt2 increasing filters numbers
        self.up1 = DeConv(327, 231)
        self.up2 = DeConv(231, 163)
        self.up3 = DeConv(163, 115)
        self.up4 = DeConv(115, 81)
        self.up5 = DeConv(81, 57)
        self.up6 = DeConv(57, 40)
        self.up7 = DeConv(40, 28)
        self.up8 = DeConv(28, 20)
        self.up9 = DeConv(20, 14)
        self.up10 = DeConv(14, 10)
        self.up11 = DeConv(10, 7)
        self.up12 = DeConv(7, 5)
        # self.outc = OutConv(5, n_classes)
        # a thought of adding one linear layer in the hidden layer:
        self.lstm = nn.LSTM(327,327,batch_first=True)
        self.mid_linear = nn.Sequential(
            #nn.Conv1d(327, 327, kernel_size = 1, stride = 1),
            nn.BatchNorm1d(327)
            )
        self.if_classifier = classifier
        self.classifier = nn.Sequential(
                nn.AvgPool1d(kernel_size=30*128, stride=30*128),
                nn.Conv1d(5, 5, kernel_size=1, stride=1),
                nn.Softmax(dim=-2)
            )
        
        
        

    def forward(self, x):

        x1 = self.inc(x)
        x2 = self.down1(nn.MaxPool1d(2,2)(x1))
        x3 = self.down2(nn.MaxPool1d(2,2)(x2))
        x4 = self.down3(nn.MaxPool1d(2,2)(x3))
        x5 = self.down4(nn.MaxPool1d(2,2)(x4))
        x6 = self.down5(nn.MaxPool1d(2,2)(x5))
        x7 = self.down6(nn.MaxPool1d(2,2)(x6))
        x8 = self.down7(nn.MaxPool1d(2,2)(x7))
        x9 = self.down8(nn.MaxPool1d(2,2)(x8))
        x10 = self.down9(nn.MaxPool1d(2,2)(x9))
        x11 = self.down10(nn.MaxPool1d(2,2)(x10))
        x12 = self.down11(nn.MaxPool1d(2,2)(x11))
        x_final = self.down12(nn.MaxPool1d(2,2)(x12))
        x_final = torch.transpose(x_final, 1,2)
        x_final, _ = self.lstm(x_final)
        x_final = torch.transpose(x_final, 1, 2)
        x_final = self.mid_linear(x_final)
        x = self.up1(x_final, x12)
        x = self.up2(x, x11)
        x = self.up3(x, x10)
        x = self.up4(x, x9)
        x = self.up5(x, x8)
        x = self.up6(x, x7)
        x = self.up7(x, x6)
        x = self.up8(x, x5)
        x = self.up9(x, x4)  
        x = self.up10(x, x3)
        x = self.up11(x, x2)
        x = self.up12(x, x1)
        
        if self.if_classifier:
            x = self.classifier(x)
        
        #logits = self.outc(x)
        
        return x