import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models

from functools import partial
from opts import _nostride2dilation
from Models.layers import CRPBlock, RCU_LW


class LightWeightRefineNet(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(LightWeightRefineNet, self).__init__()

        self.name = 'RefineNet-resnet34'
        backbone = models.resnet34(pretrained=pretrained)

        self.configs= configs

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

        # ignore the RCU_LW, all stages reduce to 128c
        # ignore conv1x1 in fusion module, using RCU instead of conv1x1
        self.layer4_pre_RCU = self._make_rcu(512, 512, nblocks=2)
        self.layer4_pre_conv1x1 = nn.Conv2d(512, 128, 1, 1, 0, bias=False)
        self.layer4_CRP = self._make_crp(128, 128, 4)
        self.layer4_post_RCU = self._make_rcu(128, 128, nblocks=3)
        self.layer4_f_conv1x1 = nn.Conv2d(128, 64, 1, 1, 0, bias=False)

        self.layer3_pre_RCU = self._make_rcu(256, 256, nblocks=2)
        self.layer3_pref_conv1x1 = nn.Conv2d(256, 64, 1, 1, 0, bias=False)
        self.layer3_CRP = self._make_crp(64, 64, 4)
        self.layer3_post_RCU = self._make_rcu(64, 64, nblocks=3)
        #self.layer3_postf_conv1x1 = nn.Conv2d(64, 64, 1, 1, 0, bias=False)

        self.layer2_pre_RCU = self._make_rcu(128, 128, nblocks=2)
        self.layer2_pref_conv1x1 = nn.Conv2d(128, 64, 1, 1, 0, bias=False)
        self.layer2_CRP = self._make_crp(64, 64, 4)
        self.layer2_post_RCU = self._make_rcu(64, 64, nblocks=3)

        self.layer1_pre_RCU = self._make_rcu(64, 64, nblocks=2)
        self.layer1_CRP = self._make_crp(64, 64, 4)
        self.layer1_post_RCU = self._make_rcu(64, 64, nblocks=3)

        self.clf_conv = nn.Conv2d(64, outc, kernel_size=3, stride=1, padding=1, bias=True)

        self._weight_init()

    def _make_rcu(self, inc, outc, nblocks):
        layers = []
        for i in range(nblocks):
            rcu = RCU_LW(inc if (i==0) else outc, outc)
            layers.append(nn.Sequential(rcu))
        return nn.Sequential(*layers)

    def _make_crp(self, inc, outc, stages):
        layers = [CRPBlock(inc, outc, stages)]
        return nn.Sequential(*layers)

    def _weight_init(self):

        for name, child in self.named_children():
            if name not in ['conv0', 'bn0', 'relu0', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']:
                for module in child.modules():
                    if isinstance(module, nn.Conv2d):
                        init.kaiming_normal_(module.weight)
                    if isinstance(module, nn.BatchNorm2d):
                        init.constant_(module.weight, 1)
                        init.constant_(module.bias, 0)


    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)   # 64
        l2 = self.layer2(l1)  # 128
        l3 = self.layer3(l2)  # 256
        l4 = self.layer4(l3)  # 512

        # dropout for l3 and l4

        x4 = self.layer4_pre_RCU(l4)
        x4 = self.layer4_pre_conv1x1(x4)
        x4 = self.layer4_CRP(x4)
        x4 = self.layer4_post_RCU(x4)
        x4 = self.layer4_f_conv1x1(x4)
        x4 = F.interpolate(x4, size=l3.size()[2:], mode='bilinear', align_corners=True)

        x3 = self.layer3_pre_RCU(l3)
        x3 = self.layer3_pref_conv1x1(x3)
        x3 = x3 + x4
        x3 = self.layer3_CRP(x3)
        x3 = self.layer3_post_RCU(x3)
        x3 = F.interpolate(x3, size=l2.size()[2:], mode='bilinear', align_corners=True)

        x2 = self.layer2_pre_RCU(l2)
        x2 = self.layer2_pref_conv1x1(x2)
        x2 = x2 + x3
        x2 = self.layer2_CRP(x2)
        x2 = self.layer2_post_RCU(x2)
        x2 = F.interpolate(x2, size=l1.size()[2:], mode='bilinear', align_corners=True)

        x1 = self.layer1_pre_RCU(l1)
        x1 = x1 + x2
        x1 = self.layer1_CRP(x1)
        x1 = self.layer1_post_RCU(x1)

        out = self.clf_conv(x1)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)

        return out





