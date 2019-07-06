import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models

from functools import partial
from opts import _nostride2dilation
from Models.layers import ASPP, _FCNHead, ConvBnRelu

class DeepLabv3_plus(nn.Module):
    def __init__(self, configs, outc=6, stride=16, pretrained=True):
        super(DeepLabv3_plus, self).__init__()

        self.name='deeplabv3_plus-resnet34'
        self.configs = configs
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

        self.low_level_adaptor = ConvBnRelu(64, 64, 1, 1, 0)
        # concat with output from aspp [128, 64]
        self.aspp_layer = ASPP(inc=512, stride=16)
        self.score_layer = _FCNHead(192, outc=outc)

    def forward(self, x):

        x = self.conv0(x)        #2
        x = self.bn0(x)          #2
        x = self.relu0(x)        #2
        x = self.maxpool(x)      #4

        low_level_fm = self.layer1(x)    #4
        x = self.layer2(low_level_fm)    #8
        x = self.layer3(x)               #16
        x = self.layer4(x)               #16

        aspp_fm = self.aspp_layer(x)
        aspp_fm = F.interpolate(aspp_fm, scale_factor=4, mode='bilinear', align_corners=True)
        low_level_fm = self.low_level_adaptor(low_level_fm)

        out = self.score_layer(torch.cat([low_level_fm, aspp_fm], dim=1))

        return F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)
