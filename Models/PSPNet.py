from functools import partial
from collections import OrderedDict

import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from Models.layers import ConvBnRelu
from opts import _nostride2dilation
# ResNet34 Based

class PSPNet(nn.Module):
    def __init__(self, configs, outc=6, stride=16, pretrained=True):
        super(PSPNet, self).__init__()

        self.name='PSPNet_resnet34'
        self.configs= configs

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


        self.business_layer = []
        self.psp_layer = PyramidPooling('psp', outc, 512,)
        '''
        self.aux_layer = nn.Sequential(
            ConvBnRelu(1024, 1024, 3, 1, 1,
                       has_bn=True,
                       has_relu=True,
                       has_bias=False)
        )
        '''

        self.business_layer.append(self.psp_layer)
        #self.business_layer.append(self.aux_layer)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)

        x = self.layer1(x) # 4
        x = self.layer2(x) # 8
        x = self.layer3(x) # 16
        x = self.layer4(x) # 16

        psp_fm = self.psp_layer(x)
        # little modification for fair comparison
        psp_fm = F.interpolate(psp_fm, scale_factor=16, mode='bilinear', align_corners=True)

        return psp_fm


class PyramidPooling(nn.Module):
    def __init__(self, name, outc, fc_dim=4096, pool_scales=[1, 2, 3, 6],):
        # fc_dim means the dim of output from backbone
        super(PyramidPooling, self).__init__()

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(OrderedDict([
                ('{}/pool_1'.format(name), nn.AdaptiveAvgPool2d(scale)),
                ('{}/cbr'.format(name),
                 # TODO here occurs error
                 ConvBnRelu(fc_dim, 128, 1, 1, 0, has_bn=True,
                            has_relu=True, has_bias=False))
            ])))

        self.ppm = nn.ModuleList(self.ppm)

        self.conv6 = nn.Sequential(
            ConvBnRelu(fc_dim + len(pool_scales) * 128, 128, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(128, outc, kernel_size=1)
        )

        self._weight_init()

    def _weight_init(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)


    def forward(self, x):
        input_size = x.size()

        ppm_out = [x]
        for pooling in self.ppm:
            tmp = pooling(x)
            ppm_out.append(
                F.interpolate(tmp, size=(input_size[2], input_size[3]),
                              mode='bilinear', align_corners=True)
            )

        ppm_out = torch.cat(ppm_out, 1)

        ppm_out = self.conv6(ppm_out)
        return ppm_out
