import sys
#!pip install './pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/'
sys.path.append('./pretrainedmodels-0.7.4/pretrainedmodels-0.7.4/')

#!pip install './efficientnet_pytorch-0.7.1/efficientnet_pytorch-0.7.1/'
sys.path.append('./efficientnet_pytorch-0.7.1/efficientnet_pytorch-0.7.1/')

import dicomsdl
def __dataset__to_numpy_image(self, index=0):
    info = self.getPixelDataInfo()
    dtype = info['dtype']
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]
    outarr = np.empty(shape, dtype=dtype)
    self.copyFrameData(index, outarr)
    return outarr
dicomsdl._dicomsdl.DataSet.to_numpy_image = __dataset__to_numpy_image   

import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import os, copy, time

import cv2
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import pydicom

import torch
from torch import nn, optim
import timm
import segmentation_models_pytorch as smp


class SegmentationModel(nn.Module):
    def __init__(self, backbone=None, segtype='unet', pretrained=False):
        super(SegmentationModel, self).__init__()
        
        n_blocks = 4
        self.n_blocks = n_blocks
        
        self.encoder = timm.create_model(
            'resnet18d',
            in_chans=3,
            features_only=True,
            drop_rate=0.1,
            drop_path_rate=0.1,
            pretrained=pretrained
        )
        g = self.encoder(torch.rand(1, 3, 64, 64))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        if segtype == 'unet':
            self.decoder = smp.decoders.unet.decoder.UnetDecoder(
                encoder_channels=encoder_channels[:n_blocks+1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )

        self.segmentation_head = nn.Conv2d(decoder_channels[n_blocks-1], 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self,x):
        x = torch.stack([x]*3, 1)
        global_features = [0] + self.encoder(x)[:self.n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features
    
# from timm.models.layers.conv2d_same import Conv2dSame
class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return timm.models.layers.conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def is_static_pad(kernel_size: int, stride: int = 1, dilation: int = 1, **_):
    return stride == 1 and (dilation * (kernel_size - 1)) % 2 == 0


def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1, 1), value: float = 0):
    ih, iw, iz = x.size()[-3:]
    pad_h = get_same_padding(ih, k[0], s[0], d[0])
    pad_w = get_same_padding(iw, k[1], s[1], d[1])
    pad_z = get_same_padding(iz, k[2], s[2], d[2])
    if pad_h > 0 or pad_w > 0 or pad_z > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2, pad_z // 2, pad_z - pad_z // 2], value=value)
    return x


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == 'same':
            if is_static_pad(kernel_size, **kwargs):
                padding = get_padding(kernel_size, **kwargs)
            else:
                padding = 0
                dynamic = True
        elif padding == 'valid':
            padding = 0
        else:
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def conv3d_same(
        x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0), dilation: Tuple[int, int, int] = (1, 1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-3:], stride, dilation)
    return F.conv3d(x, weight, bias, stride, (0, 0, 0), dilation, groups)

class Conv3dSame(nn.Conv3d):
    """ Tensorflow like 'SAME' convolution wrapper for 3d convolutions
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv3d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def create_conv3d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv3dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv3d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


def convert_3d(module):
    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.afine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))
    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))
    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )
    for name, child in module.named_children():
        module_output.add_module(
            name, convert_3d(child)
        )
    del module
    return module_output

def predict_segmentation(volumes, models):
    final_outputs = []
    for model in models:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                volumes = volumes.float() / 255
                outputs = model(volumes.cuda())
                outputs = outputs.sigmoid().detach().cpu().numpy()
                final_outputs.append(outputs)
    final_outputs = np.stack(final_outputs).mean(0)
    final_outputs = (final_outputs>0.5).astype(np.float32)
    return final_outputs

class Model(nn.Module):
    def __init__(self, pretrained=True):
        super(Model, self).__init__()
        drop = 0.2
        self.encoder = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=False, in_chans=3, global_pool='', num_classes=0, drop_rate=drop, drop_path_rate=drop)
        feats = self.encoder.num_features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        lstm_embed = 512
        self.lstm = nn.LSTM(feats, lstm_embed, num_layers=2, dropout=drop, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(nn.Linear(lstm_embed*2, lstm_embed), nn.BatchNorm1d(lstm_embed), nn.Dropout(drop), nn.LeakyReLU(0.1), nn.Linear(lstm_embed, 9))
    def forward(self, x):
        x = torch.nan_to_num(x, 0, 0, 0)
        bs, ns, c, sz, _ = x.shape
        x = x.view(bs*ns, c, sz, sz)
        feat = self.encoder(x)
        feat = self.avgpool(feat)
        feat = feat.view(bs, ns, -1)
        feat, _ = self.lstm(feat)
        feat = feat.contiguous().view(bs*ns, -1)
        feat = self.head(feat)
        return feat.view(bs, ns, -1)

class Model2(nn.Module):
    def __init__(self, pretrained=True, mask_head=False):
        super(Model2, self).__init__()
        self.mask_head = mask_head
        drop = 0.2
        true_encoder = timm.create_model('tf_efficientnetv2_s.in21k_ft_in1k', pretrained=False, in_chans=3, global_pool='', num_classes=0, drop_rate=drop, drop_path_rate=drop)
        segmentor = smp.Unet(f"tu-tf_efficientnetv2_s", encoder_weights=None, in_channels=3, classes=3)
        self.encoder = segmentor.encoder
        self.decoder = segmentor.decoder
        self.segmentation_head = segmentor.segmentation_head
        st = true_encoder.state_dict()
        self.encoder.model.load_state_dict(st, strict=False)
        self.conv_head = true_encoder.conv_head
        self.bn2 = true_encoder.bn2
        feats = true_encoder.num_features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        lstm_embed = feats * 1
        self.lstm = nn.LSTM(lstm_embed, lstm_embed//2, num_layers=2, dropout=drop, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_embed, lstm_embed//2),
            nn.BatchNorm1d(lstm_embed//2),
            nn.Dropout(drop),
            nn.LeakyReLU(0.1),
            nn.Linear(lstm_embed//2, 9),
        )
    def forward(self, x):
        x = torch.nan_to_num(x, 0, 0, 0)
        bs, n_slice_per_c, in_chans, image_size, _ = x.shape
        x = x.view(bs * n_slice_per_c, in_chans, image_size, image_size)
        features = self.encoder(x)
        if self.mask_head:
            decoded = self.decoder(*features)
            masks = self.segmentation_head(decoded)
        feat = features[-1]
        feat = self.conv_head(feat)
        feat = self.bn2(feat)
        avg_feat = self.avgpool(feat)
        avg_feat = avg_feat.view(bs, n_slice_per_c, -1)
        feat = avg_feat
        feat, _ = self.lstm(feat)
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        feat = self.head(feat)
        feat = feat.view(bs, n_slice_per_c, -1).contiguous()
        if self.mask_head:
            return feat, masks
        else:
            return feat
