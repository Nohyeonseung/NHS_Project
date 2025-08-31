import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels), # BatchNorm3d -> InstanceNorm3d
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels), # BatchNorm3d -> InstanceNorm3d
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_channels=32):
        super().__init__()
        #Incoder
        self.enc1 = DoubleConv3DBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3DBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3DBlock(base_channels * 2, base_channels * 4) 
        self.pool3 = nn.MaxPool3d(2)
        self.enc4 = DoubleConv3DBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool3d(2)

        # bottleneck
        self.bottleneck = DoubleConv3DBlock(base_channels * 8, base_channels * 16)

        #Decoder
        self.upconv4 = nn.ConvTranspose3d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv3DBlock(base_channels * 16, base_channels * 8) # in: 16*32=512, out: 8*32=256

        self.upconv3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3DBlock(base_channels * 8, base_channels * 4) # in: 8*32=256, out: 4*32=128

        self.upconv2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3DBlock(base_channels * 4, base_channels * 2) # in: 4*32=128, out: 2*32=64
        
        self.upconv1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3DBlock(base_channels * 2, base_channels)     # in: 2*32=64, out: 1*32=32

        self.final_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Incoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # bottleneck
        b = self.bottleneck(self.pool4(e4))

        #Decoder
        d4 = self.upconv4(b)
        if d4.shape != e4.shape:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='trilinear', align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='trilinear', align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return out
