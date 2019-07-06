import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from Models.base_model import resnet
from torchvision.models import vgg16_bn, vgg16, vgg11
# vgg16 based


class SegNetDec(nn.Module):
    def __init__(self, inc, outc, nb_layers):
        super(SegNetDec, self).__init__()
        layers = [
            nn.Conv2d(inc, inc // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inc // 2),
            nn.ReLU(inplace=True)
        ]
        layers += [
            nn.Conv2d(inc // 2, inc // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inc // 2),
            nn.ReLU(inplace=True)
        ] * nb_layers
        layers += [
            nn.Conv2d(inc // 2, outc, 3, padding=1, bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        ]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

class SegNet(nn.Module):
    def __init__(self, configs, outc=6, stride=32, pretrained=True):
        super(SegNet, self).__init__()

        self.name = 'SegNet-vgg11'
        backbone = vgg11(pretrained=pretrained)
        features = backbone.features
        self.enc_layer1 = features[0:6]   #2x  64c
        self.enc_layer2 = features[7:13]  #4x  128c
        self.enc_layer3 = features[14:23] #8x  256c
        self.enc_layer4 = features[24:33] #16x 512c
        self.enc_layer5 = features[34:43] #32x 512c

        self.dec_layer5 = SegNetDec(512, 512, 1)
        self.dec_layer4 = SegNetDec(512, 256, 1)
        self.dec_layer3 = SegNetDec(256, 128, 1)
        self.dec_layer2 = SegNetDec(128, 64, 0)

        self.score_layer = nn.Conv2d(64, outc, 3, 1, 1, bias=False)

        self._weight_init()

    def forward(self, x):

        x1 = self.enc_layer1(x)
        d1, m1 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        x2 = self.enc_layer2(d1)
        d2, m2 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        x3 = self.enc_layer3(d2)
        d3, m3 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        x4 = self.enc_layer4(d3)
        d4, m4 = F.max_pool2d(x4, kernel_size=2, stride=2, return_indices=True)
        x5 = self.enc_layer5(d4)
        d5, m5 = F.max_pool2d(x5, kernel_size=2, stride=2, return_indices=True)

        e5 = self.dec_layer5(F.max_unpool2d(d5, m5, kernel_size=2, stride=2, output_size=x5.size()))
        e4 = self.dec_layer4(F.max_unpool2d(e5, m4, kernel_size=2, stride=2, output_size=x4.size()))
        e3 = self.dec_layer3(F.max_unpool2d(e4, m3, kernel_size=2, stride=2, output_size=x3.size()))
        e2 = self.dec_layer2(F.max_unpool2d(e3, m2, kernel_size=2, stride=2, output_size=x2.size()))
        e1 = F.max_unpool2d(e2, m1, kernel_size=2, stride=2, output_size=x1.size())

        return self.score_layer(e1)

    def _weight_init(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight.data)