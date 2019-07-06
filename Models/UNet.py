# sub-parts of the U-Net model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models

from Models.layers import _FCNHead
from opts import _nostride2dilation
from functools import partial

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UNet_resnet(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(UNet_resnet, self).__init__()

        self.configs= configs
        self.name = 'Unet_resnet34'

        backbone = models.resnet34(pretrained=pretrained)

        self.conv0 = backbone.conv1
        self.bn0 = backbone.bn1
        self.relu0 = backbone.relu
        self.maxpool = backbone.maxpool # 64

        self.layer1 = backbone.layer1 # 64
        self.layer2 = backbone.layer2 # 128
        self.layer3 = backbone.layer3 # 256
        self.layer4 = backbone.layer4 # 512

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.requires_grad = False

        self.stride = stride
        self.spatial_size = [self.configs.size_cropped_images_h // self.stride,
                             self.configs.size_cropped_images_w // self.stride]
        if self.stride == 8:
            self.layer3.apply(partial(_nostride2dilation, dilation=2))
            self.layer4.apply(partial(_nostride2dilation, dilation=4))
        elif self.stride == 16:
            self.layer4.apply(partial(_nostride2dilation, dilation=2))

        self.layer5 = down(in_ch=512, out_ch=512) #64

        self.up4 = double_conv(1024, 256)
        self.up3 = double_conv(512, 128)
        self.up2 = double_conv(256, 64)

        self.score_layer = _FCNHead(64, outc=outc)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)    # 2  64

        x1 = self.layer1(x)  # 2  64
        x2 = self.layer2(x1) # 4  128
        x3 = self.layer3(x2) # 8  256
        x4 = self.layer4(x3) # 16 512
        x5 = self.layer5(x4) # 32 512

        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)
        x4 = torch.cat([x4, x5], dim=1)
        x4 = self.up4(x4) # 256
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = torch.cat([x3, x4], dim=1)
        x3 = self.up3(x3) # 128
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = torch.cat([x2, x3], dim=1)
        x2 = self.up2(x2) # 64

        out = self.score_layer(x2)
        return F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)


