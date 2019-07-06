import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from functools import partial

from opts import _nostride2dilation
from Models.layers import ASPP, _FCNHead

class DeepLabv3(nn.Module):
    def __init__(self, configs, outc=6, stride=16, pretrained=True):
        super(DeepLabv3, self).__init__()

        self.configs = configs
        self.name='DeepLabv3-resnet34'

        backbone = models.resnet34(pretrained=pretrained)

        self.conv0 = backbone.conv1
        self.bn0 = backbone.bn1
        self.relu0 = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

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

        # after con1x1
        self.aspp = ASPP(inc = 512, stride=stride) # out 128c

        # classifier
        self.score_layer = _FCNHead(128, outc=outc)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.backbone(x)
        x = self.aspp(x)
        x = self.score_layer(x)
        # little modification for efficiency
        return F.interpolate(x, scale_factor=16, mode='bilinear', align_corners=True)
