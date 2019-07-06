import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils import model_zoo
from torchvision import models
from torchvision.models import resnet34,resnet101
from Models.base_model import resnet
from Models.layers import ConvBnRelu
from functools import partial
from opts import _nostride2dilation
from Models.layers import _FCNHead

class FCN8s(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(FCN8s, self).__init__()

        self.name='FCN8s_resnet34'
        self.configs=configs
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

        self.score_layer4 = _FCNHead(512, outc=outc)
        self.score_layer3 = _FCNHead(256, outc=outc)
        self.score_layer2 = _FCNHead(128, outc=outc)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        score_x2 = self.score_layer2(x2)
        x3 = self.layer3(x2)
        score_x3 = self.score_layer3(x3)

        x4 = self.layer4(x3)
        score_x4 = self.score_layer4(x4)
        score_x4 = F.interpolate(score_x4, size=score_x3.size()[2:],mode='bilinear',align_corners=True)
        score_x3 += score_x4
        score_x3 = F.interpolate(score_x3, size=score_x2.size()[2:],mode='bilinear', align_corners=True)
        score_x2 += score_x3
        return F.interpolate(score_x2, size=x.size()[2:], mode='bilinear', align_corners=True)

class FCN16s(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(FCN16s, self).__init__()

        self.name='FCN16s_resnet34'
        self.configs=configs
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

        self.score_layer4 = _FCNHead(512, outc=outc)
        self.score_layer3 = _FCNHead(256, outc=outc)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        score_x3 = self.score_layer3(x3)
        score_x4 = self.score_layer4(x4)
        score_x4 = F.interpolate(score_x4, scale_factor=2, mode='bilinear', align_corners=True)

        score_x3 = score_x3 + score_x4

        return F.interpolate(score_x3, scale_factor=16, mode='bilinear', align_corners=True)

class FCN32s(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(FCN32s, self).__init__()

        self.name='FCN32s_resnet34'
        self.configs=configs
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

        self.score_layer4 = _FCNHead(512, outc=outc)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        score_x4 = self.score_layer4(x4)

        return F.upsample_bilinear(score_x4, x.size()[2:])

class FCN_atrous_network(nn.Module):
    def __init__(self, configs, outc=6, stride=8, pretrained=True):
        super(FCN_atrous_network, self).__init__()

        self.configs = configs
        self.name = 'FCN_atrous_network'

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

        self.score_layer = _FCNHead(inc=512, outc=outc)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.score_layer(x)

        return F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

# ResNet101-based
class FCN8s_resnet101(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(FCN8s_resnet101, self).__init__()

        self.name = 'FCN8s_resnet101'
        self.configs= configs

        backbone = resnet101(pretrained=pretrained)
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


        self.score_layer4 = _FCNHead(2048, outc=outc)
        self.score_layer3 = _FCNHead(1024, outc=outc)
        self.score_layer2 = _FCNHead(512, outc=outc)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpooling(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        score_x2 = self.score_layer2(x2)
        score_x3 = self.score_layer3(x3)
        score_x4 = self.score_layer4(x4)

        score_x4 = F.interpolate(score_x4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        score_x3 = score_x3 + score_x4
        score_x3 = F.interpolate(score_x3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        score_x2 = score_x2 + score_x3

        return F.interpolate(score_x2, scale_factor=8, mode='bilinear', align_corners=True)
