import torch
import torch. nn as nn
import numpy
import torch.nn.functional as F

# class FirstConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.First_conv = nn.Sequential(
#             nn.MaxPool1d(16),
#             nn.Conv1d(in_channels, out_channels, kernel_size=128, stride=1, padding='same', bias=False),
#         )

#     def forward(self, x):
#         return self.First_conv(x)

class EnConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.en_conv = nn.Sequential(

            nn.Conv1d(in_channels, mid_channels, kernel_size = 9, stride = 1, padding='same'),
            nn.ELU(), 
            nn.Dropout(p=0.01),
            nn.BatchNorm1d(mid_channels),

        )

    def forward(self, x):
        return self.en_conv(x)

class DeConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.de_conv = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Conv1d(in_channels, mid_channels, kernel_size = 2, stride = 1, padding='same'),
            nn.ELU(), 
            nn.Dropout(p=0.01),
            nn.BatchNorm1d(mid_channels)         
        )
        self.out_conv = nn.Sequential( 
            nn.Conv1d(mid_channels*2, mid_channels, kernel_size = 2, stride = 1, padding='same'),
            nn.ELU(), 
            nn.BatchNorm1d(mid_channels)         
        )

    def forward(self, x1, x2):
        
        x1 = self.de_conv(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff//2, diff-diff//2])
        x = torch.cat([x2, x1], dim=1)
        return self.out_conv(x)





class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.avgpool = nn.AvgPool1d(kernel_size=30*128, stride=30*128)
        self.conv1 = nn.Conv1d(in_channels,in_channels,1, 1)
        self.elu = nn.ELU()
        self.batchnorm = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels,out_channels,1, 1)
        self.sftmax = nn.Softmax(dim=-2)
        self.dp = nn.Dropout(p=0.01)
        
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = self.elu(x)
        x = self.dp(x)
        x = self.batchnorm(x)
        x = self.conv2(x)
        return self.sftmax(x)