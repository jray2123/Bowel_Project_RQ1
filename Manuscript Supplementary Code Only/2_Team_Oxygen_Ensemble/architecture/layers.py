"""
Compatibility module for CoaT model imports
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# PixelShuffle_ICNR implementation
class PixelShuffle_ICNR(nn.Module):
    def __init__(self, ni, nf, scale=2, blur=False):
        super().__init__()
        # Use nn.Sequential for shuf
        self.shuf = nn.Sequential(
            nn.Conv2d(ni, nf * (scale ** 2), kernel_size=1),
            nn.Conv2d(nf * (scale ** 2), nf * (scale ** 2), kernel_size=1)
        )
        self.pixel_shuffle = nn.PixelShuffle(scale)
        # ICNR init
        nn.init.kaiming_normal_(self.shuf[0].weight)
        self.shuf[0].weight.data.copy_(self._icnr_init(self.shuf[0].weight.data))
        nn.init.kaiming_normal_(self.shuf[1].weight)
        self.blur = blur
        if blur:
            self.blur_layer = nn.AvgPool2d(2, stride=1, padding=1)

    def _icnr_init(self, x):
        ni, nf, h, w = x.shape
        ni2 = int(ni / 4)
        k = math.sqrt(1 / ni2)
        x = x.new_zeros((ni2, nf, h, w))
        for i in range(ni2):
            x[i] = nn.init.uniform_(x[i], -k, k)
        x = x.repeat(4, 1, 1, 1)
        return x

    def forward(self, x):
        x = self.shuf(x)
        x = self.pixel_shuffle(x)
        if self.blur:
            x = self.blur_layer(x)
        return x

# LSTM block used in the model
class LSTM_block(nn.Module):
    def __init__(self, inp):
        super().__init__()
        # Based on checkpoint keys: lstm.0.lstm.weight_ih_l0, lstm.0.lstm.weight_hh_l0, 
        # lstm.0.lstm.bias_ih_l0, lstm.0.lstm.bias_hh_l0
        # No reverse direction keys observed, so use bidirectional=False
        self.lstm = nn.LSTM(inp, inp, num_layers=1, bidirectional=False, batch_first=True)
        # No linear layer keys observed in checkpoint
        
    def forward(self, x):
        h, _ = self.lstm(x.flatten(2).transpose(1,2))
        # Return without applying linear layer
        return x + h.transpose(1,2).reshape(x.shape)

# UnetBlock for upsampling in segmentation networks
class UnetBlock(nn.Module):
    def __init__(self, up_in_c, x_in_c, out_channels, upsample=None, use_bn=False, use_attention=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') if upsample is None else upsample
        # Set shuf_nf to match checkpoint
        if up_in_c == 512:  # dec4
            shuf_nf = 1024
            bn_c = 320
            conv1_out = 384
            conv1_in = 320 + 256
        elif up_in_c == 384:  # dec3
            shuf_nf = 768
            bn_c = 256
            conv1_out = 192
            conv1_in = 256 + 192
        elif up_in_c == 192:  # dec2
            shuf_nf = 384
            bn_c = 128
            conv1_out = 96
            conv1_in = 128 + 96
        else:
            shuf_nf = out_channels * 4
            bn_c = out_channels // 2
            conv1_out = out_channels
            conv1_in = bn_c + x_in_c
        self.shuf = nn.Sequential(
            nn.Conv2d(up_in_c, shuf_nf, kernel_size=1, bias=True),
            nn.BatchNorm2d(shuf_nf, track_running_stats=False)
        )
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.bn = nn.BatchNorm2d(bn_c, track_running_stats=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv1_in, conv1_out, kernel_size=3, padding=1, bias=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_out, conv1_out, kernel_size=3, padding=1, bias=True)
        )
    def forward(self, up_in, x_in):
        up_out = self.upsample(up_in)
        up_out = self.shuf(up_out)
        up_out = self.pixel_shuffle(up_out)
        up_out = F.relu(self.bn(up_out))
        cat_x = torch.cat([up_out, x_in], dim=1)
        x = F.relu(self.conv1(cat_x))
        x = F.relu(self.conv2(x))
        return x

# FPN module for feature pyramid networks
class FPN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.convs = nn.ModuleList()
        for in_c, out_c in zip(input_channels, output_channels):
            # conv→BN→conv matching checkpoint keys 0,2,3
            # conv->BN->conv: final conv outputs out_c//2 channels to match checkpoint
            self.convs.append(nn.Sequential(OrderedDict([
                ('0', nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=True)),
                ('2', nn.BatchNorm2d(out_c, track_running_stats=False)),
                ('3', nn.Conv2d(out_c, out_c//2, kernel_size=3, padding=1, bias=True))
            ])))
    def forward(self, xs, last_layer):
        return [conv(x) for conv, x in zip(self.convs, xs)]

# UpBlock for upsampling 
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blur=False):
        super().__init__()
        self.blur = blur
        # Special-case for final_conv
        if in_channels == 192 and out_channels == 192:
            shuf_nf = 192
            scale = 2
            self.shuf = nn.Sequential(
                nn.Conv2d(in_channels, shuf_nf, kernel_size=1, bias=True),
                nn.BatchNorm2d(shuf_nf, track_running_stats=False)
            )
            self.pixel_shuffle = nn.PixelShuffle(scale)
            self.conv = nn.Sequential(OrderedDict([
                ('0', nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=True)),
                ('1', nn.BatchNorm2d(48, track_running_stats=False)),
                ('3', nn.Conv2d(48, 4, kernel_size=1, bias=True))
            ]))
            self.out_proj = nn.Identity()
        else:
            shuf_nf = out_channels * 4
            scale = 2
            self.shuf = nn.Sequential(
                nn.Conv2d(in_channels, shuf_nf, kernel_size=1, bias=True),
                nn.BatchNorm2d(shuf_nf)
            )
            self.pixel_shuffle = nn.PixelShuffle(scale)
            self.conv = nn.Sequential(OrderedDict([
                ('0', nn.Conv2d(shuf_nf // (scale * scale), shuf_nf // (scale * scale), kernel_size=3, padding=1, bias=True)),
                ('2', nn.BatchNorm2d(shuf_nf // (scale * scale))),
                ('3', nn.ReLU(inplace=True))
            ]))
            self.out_proj = nn.Identity()
    def forward(self, x):
        x = self.shuf(x)
        x = self.pixel_shuffle(x)
        x = self.conv(x)
        x = self.out_proj(x)
        return x