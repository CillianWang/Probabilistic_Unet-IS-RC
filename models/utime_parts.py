import torch
import torch.nn as nn
import torch.nn.functional as F

class FirstConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.First_conv = nn.Sequential(
            
            nn.MaxPool1d(16),
            nn.Conv1d(in_channels, out_channels, kernel_size=128, stride=1, bias=False),
        )

    def forward(self, x):
        return self.First_conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=5, stride = 1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]


        x1 = F.pad(x1, [diff//2, diff-diff//2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up1 = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=8, stride=2)
        
        self.up2 = nn.ConvTranspose1d(in_channels//2, in_channels // 4, kernel_size=8, stride=2)
        self.up_to_out = nn.ConvTranspose1d(in_channels //4, in_channels //8, kernel_size=8, stride = 4)
        self.conv = nn.Conv1d(in_channels//8, out_channels, kernel_size=1)
        
    def forward(self, x,length):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up_to_out(x)
        diff = length-x.shape[-1]
        x = F.pad(x, [diff//2, diff-diff//2])
        return self.conv(x)