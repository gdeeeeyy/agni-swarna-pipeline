import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if necessary
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SmallUNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_c=64):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_c)
        self.enc2 = ConvBlock(base_c, base_c*2)
        self.enc3 = ConvBlock(base_c*2, base_c*4)
        self.bottom = ConvBlock(base_c*4, base_c*8)

        self.up1 = UpBlock(base_c*8, base_c*4, base_c*4)
        self.up2 = UpBlock(base_c*4, base_c*2, base_c*2)
        self.up3 = UpBlock(base_c*2, base_c, base_c)

        self.final = nn.Conv2d(base_c, out_ch, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        xb = self.bottom(self.pool(x3))
        x = self.up1(xb, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.final(x)
