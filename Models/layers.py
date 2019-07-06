from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ConvBnRelu(nn.Module):
    def __init__(self, inc, outc, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 bn_eps=1e-5, has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size=ksize, stride=stride,
                              padding=pad, dilation=dilation, groups=groups,
                              bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(outc, eps=bn_eps)

        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

        self._weight_init()

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)

        if self.has_relu:
            x = self.relu(x)

        return x
    def _weight_init(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)


class Depthwise_Separable_ConvBnReLU(nn.Module):
    def __init__(self, inc, outc, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 bn_eps=1e-5, has_relu=True, inplace=True, has_bias=False):
        super(Depthwise_Separable_ConvBnReLU, self).__init__()
        # outc == inc by default
        # outc % inc ==0 / inc * exploration = outc

        assert outc % groups == 0, \
            'outc % groups must be 0 in depthwise convolution !'

        self.conv_depth = nn.Conv2d(inc, outc, ksize, stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        self.conv_point = nn.Conv2d(outc, outc, 1, 1, 0, bias=False)
        self.has_relu = has_relu
        if self.has_bn:
            self.bn_after_depth = norm_layer(outc, eps=bn_eps)
            self.bn_after_point = norm_layer(outc, eps=bn_eps)
        if self.has_relu:
            self.relu_after_point = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv_depth(x)
        x = self.bn_after_depth(x)
        x = self.conv_point(x)
        x = self.bn_after_point(x)

        return x

# for light-weight-refinenet
class CRPBlock(nn.Module):
    def __init__(self, inc, outc, nstages):
        # chain residual pooling
        super(CRPBlock, self).__init__()
        for i in range(nstages):
            setattr(self, '{}_{}'.format(i+1, 'outvar_dimred'),
                    nn.Conv2d(inc if (i==0) else outc, outc, kernel_size=1,
                              stride=1, padding=0,bias=False))

        self.stride = 1
        self.nstages = nstages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.nstages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i+1, 'outvar_dimred'))(top)
            x = top + x
        return x

class RCU_LW(nn.Module):
    def __init__(self, inc, outc):
        super(RCU_LW, self).__init__()

        self.inner_channels = inc // 4
        self.relu0 = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(inc, self.inner_channels,1,1,0,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inner_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(self.inner_channels, self.inner_channels,
                               3,1,1,bias=False)
        self.bn2 = nn.BatchNorm2d(self.inner_channels)
        self.relu3 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(self.inner_channels, outc, 1, 1, 0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(outc)

    def forward(self, x):
        input = x
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.conv3(x)
        x = self.bn3(x)

        out = x + input
        return out

# for deeplabv3
class _ASPPModule(nn.Module):
    def __init__(self, inc, outc, kernel_size, padding,
                 dilation, BatchNorm=nn.BatchNorm2d):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inc, outc,
                                     kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation)
        self.bn = BatchNorm(outc)
        self.relu = nn.ReLU()

        self._weight_init()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _weight_init(self, ):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

class ASPP(nn.Module):
    def __init__(self, inc, stride):
        super(ASPP, self).__init__()

        # inc from dfc_network
        # stride from dfc_network

        # TODO dilation need to be confirmed
        # TODO here modification to adapt the size of input
        # version1 [1,6,12,18]
        if stride == 16:
            self.dilation = [1, 6, 12, 18]
        elif stride == 8:
            self.dilation = [1, 6, 12, 18]

        self.inner_channel = inc // len(self.dilation)

        self.aspp1 = _ASPPModule(inc, self.inner_channel,
                                 kernel_size = 1, padding = 0, dilation = self.dilation[0])
        self.aspp2 = _ASPPModule(inc, self.inner_channel,
                                 kernel_size = 3, padding = self.dilation[1], dilation=self.dilation[1])
        self.aspp3 = _ASPPModule(inc, self.inner_channel,
                                 kernel_size = 3, padding = self.dilation[2], dilation=self.dilation[2])
        self.aspp4 = _ASPPModule(inc, self.inner_channel,
                                 kernel_size = 3, padding = self.dilation[3], dilation=self.dilation[3])

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inc, self.inner_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.inner_channel),
            nn.ReLU()
        )
        self.conv = nn.Conv2d(640, self.inner_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(self.inner_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.5)

        self._weight_init()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)

        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return self.dropout(x)

    def _weight_init(self):

        for name, child in self.named_children():
            if name not in ['aspp1', 'aspp2', 'aspp3', 'aspp4']:
                for submodule in child.modules():
                    if isinstance(submodule, nn.Conv2d):
                        init.kaiming_normal_(submodule.weight)
                    if isinstance(submodule, nn.BatchNorm2d):
                        init.constant_(submodule.weight, 1)
                        init.constant_(submodule.bias, 0)


# for depth-wise separable conv ASPP
class _depthwise_ASPPModule(nn.Module):
    def __init__(self, inc, outc, kernel_size, padding,
                 dilation, groups, BatchNorm=nn.BatchNorm2d):
        super(_depthwise_ASPPModule, self).__init__()
        # default

        self.atrous_depthwise_ConvBnReLU = Depthwise_Separable_ConvBnReLU(
            inc = inc, outc = outc, ksize=kernel_size,
            stride=1, pad = padding, dilation=dilation, groups=groups
        )
    def forward(self, x):

        return self.atrous_depthwise_ConvBnReLU(x)


class Depthwise_ASPP(nn.Module):
    def __init__(self, inc, stride, BatchNorm=nn.BatchNorm2d):
        super(Depthwise_ASPP, self).__init__()

        self.dilation = [1, 6, 12, 18]

        self.inner_channel = inc // len(self.dilation)

        self.aspp1 = _depthwise_ASPPModule(inc, self.inner_channel, kernel_size=1, padding=0,
                                           dilation=self.dilation[0], groups=self.inner_channel)
        self.aspp2 = _depthwise_ASPPModule(inc, self.inner_channel, kernel_size=3, padding=self.dilation[1],
                                           dilation=self.dilation[1], groups=self.inner_channel)
        self.aspp3 = _depthwise_ASPPModule(inc, self.inner_channel, kernel_size=3, padding=self.dilation[2],
                                           dilation=self.dilation[2], groups=self.inner_channel)
        self.aspp4 = _depthwise_ASPPModule(inc, self.inner_channel, kernel_size=3, padding=self.dilation[3],
                                           dilation=self.dilation[3], groups=self.inner_channel)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # TODO modification, cancel groups in 1x1 proj_conv
            nn.Conv2d(inc, self.inner_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(self.inner_channel),
            nn.ReLU()
        )
        # here no depthwise
        self.conv = nn.Conv2d(self.inner_channel * (len(self.dilation) + 1), self.inner_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(self.inner_channel)
        self.relu = nn.ReLU()
        # TODO need to be confirmed
        self.dropout = nn.Dropout2d(p=0.5)

    def forward(self, x):

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)

        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return self.dropout(x)

# for sampling layer
class Sampling_Module(nn.Module):
    def __init__(self, inc):
        '''
        :return weight sampling map, adding sigmoid layer
        '''
        super(Sampling_Module, self).__init__()

        self.sampling_conv = self._make_sampling_conv(inc=inc)
        self._weight_init()
    def forward(self, input):

        sampling_map = self._sampling_map_init(batch_size=input.shape[0], spatial_size=input.shape[2:], device=input.device)
        sampling_map = sampling_map + self.sampling_conv(input)
        sampling_map = torch.clamp(sampling_map, -1, 1)
        sampling_fm = F.grid_sample(input, sampling_map.permute(0, 2, 3, 1))
        assert sampling_fm.shape == input.shape, \
            'shape error of sampling_fm, {}'.format(sampling_fm.shape)
        # TODO change aggregation scheme: element-wise addition
        # TODO change the aggregation scheme: residual learning
        # TODO change return scheme
        # shape [batch_size, channels, H, W]
        sampling_weight_fm = torch.sigmoid(sampling_fm)
        #sampling_weight_fm = torch.sigmoid(sampling_fm)
        #return sampling_fm
        # TODO ablations based on CAM_SAM_2
        # TODO sampling_map
        # TODO sigmoid + multi
        # TODO sigmoid + multi + input without fusion + input
        return input + input * sampling_weight_fm

    def _make_sampling_conv(self, inc):
        inter_channel = inc // 2
        return nn.Sequential(
            nn.Conv2d(inc, inter_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(),
            nn.Conv2d(inter_channel, 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2)
        )

    def _sampling_map_init(self, batch_size, spatial_size, device):
        # batch_size for the last iteration of each epoch, when the total size of dataset cannot be divided by batch_size
        x = torch.linspace(-1, 1, spatial_size[0], device=device)
        y = torch.linspace(-1, 1, spatial_size[1], device=device)
        xv, yv = torch.meshgrid(x, y)
        xyv = torch.stack([xv, yv], dim=0)
        return xyv.repeat(batch_size, 1, 1, 1)

    def _weight_init(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.normal_(module.weight, mean=0, std=0.001)
            if isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)
# classifier, score_layer
class _FCNHead(nn.Module):
    def __init__(self, inc, outc, inplace=True, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        interc = inc // 4
        self.cbr = ConvBnRelu(inc, interc, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, inplace=inplace, has_bias=False)
        #self.dropout = nn.Dropout2d(0.1)
        self.conv1x1 = nn.Conv2d(interc, outc, kernel_size=1, stride=1, padding=0)
        self._weight_init()

    def forward(self, x):
        x = self.cbr(x)
        #x = self.dropout(x)
        x = self.conv1x1(x)
        return x

    def _weight_init(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

# for gru fusion
class channel_adaptive_layer(nn.Module):
    def __init__(self, inc, outc):
        super(channel_adaptive_layer, self).__init__()

        self.conv1 = nn.Conv2d(inc, outc, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.gav = nn.AdaptiveAvgPool2d(1)

        self._weight_init()

    def forward(self, x):

        pre_x = self.conv1(x)
        x = self.bn1(pre_x)
        x = self.gav(x)

        return pre_x, torch.sigmoid(x)

    def _weight_init(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

class GRU_Fusion_Module(nn.Module):
    def __init__(self, gru_input_dim, gru_hidden_dim, bias=False,
                 cell_num=1, bidirectional=False):
        super(GRU_Fusion_Module, self).__init__()

        self.GRU = nn.GRU(input_size=gru_input_dim, hidden_size=gru_hidden_dim, num_layers=cell_num,
                          bias=bias, bidirectional=bidirectional)

        #self.refine_conv1x1 = nn.Conv2d()

    def forward(self, information_map_list, h0):

        input = []
        for information_map in information_map_list:
            input.append(information_map.view(information_map.shape[0], -1))

        # shape [seq_len, batch_size, dim]
        input = torch.stack(input, dim=0)
        assert input.shape == torch.Size((len(information_map_list), h0.shape[0], h0.shape[1])), \
            'shape error of gru input, shape {}'.format(input.shape)

        h0 = h0.view(h0.shape[0], -1).unsqueeze(0)
        hidden_states, _ = self.GRU(input, h0)
        hidden_states = torch.cat([h0, hidden_states], dim=0) # shape as [3, 16, 128]

        assert hidden_states.shape == torch.Size((len(information_map_list)+1, h0.shape[1], h0.shape[2])), \
            'shape error of hidden_states, shape {}'.format(hidden_states.shape)

        hidden_states = F.softmax(hidden_states, dim=0)

        return hidden_states

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

        self._weight_init()

    def forward(self, x):
        x = self.conv(x)
        return x

    def _weight_init(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            if isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

# for ablation studies
class SAM_counterpart(nn.Module):
    def __init__(self, inc):
        super(SAM_counterpart, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(inc, inc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inc),
            nn.ReLU(),
        )

    def forward(self, x):

        return torch.sigmoid(self.conv_layer(x))
