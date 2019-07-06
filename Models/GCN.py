import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models

from Models.base_model import resnet
from opts import _nostride2dilation
from functools import partial

class GCN_module(nn.Module):
    def __init__(self, inc, outc, ks=7):
        super(GCN_module, self).__init__()

        self.conv_l1 = nn.Conv2d(inc, outc, kernel_size=(ks, 1),
                                 padding=(ks//2, 0))

        self.conv_l2 = nn.Conv2d(outc, outc, kernel_size=(1, ks),
                                 padding=(0, ks//2))
        self.conv_r1 = nn.Conv2d(inc, outc, kernel_size=(1, ks),
                                 padding=(0, ks//2))
        self.conv_r2 = nn.Conv2d(outc, outc, kernel_size=(ks, 1),
                                 padding=(ks//2, 0))
    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        return x_l + x_r

class Refine(nn.Module):
    def __init__(self, outc):
        super(Refine, self).__init__()

        self.bn = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(outc, outc, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(outc, outc, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)

        out = residual + x
        return out

class GCN(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(GCN, self).__init__()

        self.name = 'GCN_resnet34'
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

        self.gcn1 = GCN_module(512, outc)
        self.gcn2 = GCN_module(256, outc)
        self.gcn3 = GCN_module(128, outc)
        self.gcn4 = GCN_module(64, outc)

        self.refine1 = Refine(outc)
        self.refine2 = Refine(outc)
        self.refine3 = Refine(outc)
        self.refine4 = Refine(outc)
        self.refine5 = Refine(outc)
        self.refine6 = Refine(outc)
        self.refine7 = Refine(outc)
        self.refine8 = Refine(outc)
        self.refine9 = Refine(outc)

        self._weight_init()

    def forward(self, x):

        #input = x
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        # conv_x = x
        x = self.maxpool(x)
        # pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        # ->gc_module->refine_module
        gcfm1 = self.refine1(self.gcn1(fm4))
        gcfm2 = self.refine2(self.gcn2(fm3))
        gcfm3 = self.refine3(self.gcn3(fm2))
        gcfm4 = self.refine4(self.gcn4(fm1))

        # interpolate / deconvolution
        fs1 = self.refine5(F.interpolate(gcfm1, fm3.size()[2:], mode='bilinear', align_corners=True) + gcfm2)
        fs2 = self.refine6(F.interpolate(fs1, fm2.size()[2:], mode='bilinear', align_corners=True) + gcfm3)
        fs3 = self.refine7(F.interpolate(fs2, fm1.size()[2:], mode='bilinear', align_corners=True) + gcfm4)

        out = self.refine8(F.interpolate(fs3, scale_factor=2, mode='bilinear', align_corners=True))
        out = self.refine9(F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True))

        return out

    def _weight_init(self):

        for name, child in self.named_children():
            if name not in ['conv0', 'bn0', 'relu0', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']:
                for module in child.modules():
                    if isinstance(module, nn.Conv2d):
                        init.kaiming_normal_(module.weight)
                    if isinstance(module, nn.BatchNorm2d):
                        init.constant_(module.weight, 1)
                        init.constant_(module.bias, 0)