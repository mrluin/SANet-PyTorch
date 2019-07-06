import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from functools import partial
from opts import _nostride2dilation
from Models.layers import _FCNHead, Sampling_Module
from Models.layers import SAM_counterpart


class FCN8s_resnet34(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(FCN8s_resnet34, self).__init__()

        self.name = 'ablation_FCN8s'
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

        self.score_layer = _FCNHead(128, outc=outc)

        self.ada_layer4 = nn.Conv2d(512, 128, 1, 1, 0, bias=False)
        self.ada_layer3 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.ada_layer2 = nn.Conv2d(128, 128, 1, 1, 0, bias=False)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.ada_layer4(x4)
        x4 = F.interpolate(x4, scale_factor=4, mode='bilinear', align_corners=True)
        x3 = self.ada_layer3(x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = x2 + x3 + x4
        score = self.score_layer(x2)
        return F.interpolate(score, scale_factor=8, mode='bilinear', align_corners=True)


class FCN8s_SAM_M(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(FCN8s_SAM_M, self).__init__()
        # SAM -> fusion -> socre_layer
        self.name = 'FCN8s_SAM_M'
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

        self.score_layer = _FCNHead(128, outc=outc)

        self.ada_layer4 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.ada_layer3 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.ada_layer2 = nn.Conv2d(128, 128, 1, 1, 0, bias=False)

        self.sampling_conv1 = Sampling_Module(128)
        self.sampling_conv2 = Sampling_Module(256)
        self.sampling_conv3 = Sampling_Module(512)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.ada_layer4(self.sampling_conv3(x4))
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = x3 + x4
        x3 = self.ada_layer3(self.sampling_conv2(x3))
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = x2 + x3
        x2 = self.ada_layer2(self.sampling_conv1(x2))

        score = self.score_layer(x2)

        return F.interpolate(score, scale_factor=8, mode='bilinear', align_corners=True)

class FCN8s_SAM_M_resnet101(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(FCN8s_SAM_M_resnet101, self).__init__()
        # SAM -> fusion -> score_layer
        self.name = 'FCN8s_SAM_M_resnet101'
        self.configs = configs

        backbone = models.resnet101(pretrained=pretrained)

        self.conv0 = backbone.conv1
        self.bn0 = backbone.bn1
        self.relu0 = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1 # 256
        self.layer2 = backbone.layer2 # 512
        self.layer3 = backbone.layer3 # 1024
        self.layer4 = backbone.layer4 # 2048

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

        self.score_layer = _FCNHead(512, outc=outc)

        self.ada_layer4 = nn.Conv2d(2048, 1024, 1, 1, 0, bias=False)
        self.ada_layer3 = nn.Conv2d(1024, 512, 1, 1, 0, bias=False)
        self.ada_layer2 = nn.Conv2d(512, 512, 1, 1, 0, bias=False)

        self.sampling_conv1 = Sampling_Module(512)
        self.sampling_conv2 = Sampling_Module(1024)
        self.sampling_conv3 = Sampling_Module(2048)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.ada_layer4(self.sampling_conv3(x4))
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = x3 + x4
        x3 = self.ada_layer3(self.sampling_conv2(x3))
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = x2 + x3
        x2 = self.ada_layer2(self.sampling_conv1(x2))

        score = self.score_layer(x2)

        return F.interpolate(score, scale_factor=8, mode='bilinear', align_corners=True)

class FCN8s_SAM_S(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(FCN8s_SAM_S, self).__init__()
        # fusion -> SAM -> score_layer
        self.name = 'FCN8s_SAM_S'
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

        self.score_layer = _FCNHead(128, outc=outc)

        self.ada_layer4 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.ada_layer3 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.ada_layer2 = nn.Conv2d(128, 128, 1, 1, 0, bias=False)

        self.sampling_conv1 = Sampling_Module(128)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.ada_layer4(x4)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = x3 + x4
        x3 = self.ada_layer3(x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = x2 + x3
        x2 = self.ada_layer2(x2)
        x2 = self.sampling_conv1(x2)
        score = self.score_layer(x2)

        return F.interpolate(score, scale_factor=8, mode='bilinear', align_corners=True)

class FCN8s_SAM_SC(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(FCN8s_SAM_SC, self).__init__()
        # fusion -> spatial attention module -> score_layer
        self.name = 'FCN8s_SAM_2_2_3_1_counterpart'
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

        self.score_layer = _FCNHead(128, outc=outc)

        self.ada_layer4 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.ada_layer3 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.ada_layer2 = nn.Conv2d(128, 128, 1, 1, 0, bias=False)

        self.sa_counter_conv1 = SAM_counterpart(128)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.ada_layer4(x4)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = x3 + x4
        x3 = self.ada_layer3(x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = x2 + x3
        x2 = self.ada_layer2(x2)
        sa_counter = self.sa_counter_conv1(x2)
        x2 = x2 * sa_counter
        score = self.score_layer(x2)

        return F.interpolate(score, scale_factor=8, mode='bilinear', align_corners=True)

class FCN8s_SAM_MS(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(FCN8s_SAM_MS, self).__init__()
        # spatial attention module -> fusion -> score_layer
        self.name = 'FCN8s_SAM_MS'
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

        self.score_layer = _FCNHead(128, outc=outc)

        self.ada_layer4 = nn.Conv2d(512, 256, 1, 1, 0, bias=False)
        self.ada_layer3 = nn.Conv2d(256, 128, 1, 1, 0, bias=False)
        self.ada_layer2 = nn.Conv2d(128, 128, 1, 1, 0, bias=False)

        self.sampling_conv1 = Sampling_Module(128)
        self.sampling_conv2 = Sampling_Module(256)
        self.sampling_conv3 = Sampling_Module(512)

        self.sa_counter4 = SAM_counterpart(512)
        self.sa_counter3 = SAM_counterpart(256)
        self.sa_counter2 = SAM_counterpart(128)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        sa_map4 = self.sa_counter4(x4)
        x4 = self.ada_layer4(x4 * sa_map4)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x3 = x3 + x4
        sa_map3 = self.sa_counter3(x3)
        x3 = self.ada_layer3(x3 * sa_map3)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=True)
        x2 = x2 + x3
        sa_map2 = self.sa_counter2(x2)
        x2 = self.ada_layer2(x2 * sa_map2)

        score = self.score_layer(x2)

        return F.interpolate(score, scale_factor=8, mode='bilinear', align_corners=True)
