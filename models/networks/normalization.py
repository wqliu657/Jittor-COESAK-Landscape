"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
from jittor import init
import re
import jittor.nn as nn
from models.networks.spectral_norm import spectral_norm

def get_nonspade_norm_layer(opt, norm_type='instance'):

    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.shape[0]

    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        if ((subnorm_type == 'none') or (len(subnorm_type) == 0)):
            return layer
        if (getattr(layer, 'bias', None) is not None):
            delattr(layer, 'bias')
            setattr(layer, 'bias', None)
            # layer.register_parameter('bias', None)
        if (subnorm_type == 'batch'):
            norm_layer = nn.BatchNorm(get_out_channel(layer), affine=True)
        elif (subnorm_type == 'sync_batch'):
            norm_layer = nn.BatchNorm(get_out_channel(layer), affine=True)
        elif (subnorm_type == 'instance'):
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError(('normalization layer %s is not recognized' % subnorm_type))
        return nn.Sequential(layer, norm_layer)
    return add_norm_layer

class SPADE(nn.Module):

    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()
        assert config_text.startswith('spade')
        parsed = re.search('spade(\\D+)(\\d)x\\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))
        if (param_free_norm_type == 'instance'):
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif (param_free_norm_type == 'syncbatch'):
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        elif (param_free_norm_type == 'batch'):
            self.param_free_norm = nn.BatchNorm(norm_nc, affine=False)
        else:
            raise ValueError(('%s is not a recognized param-free norm type in SPADE' % param_free_norm_type))
        nhidden = 128
        pw = (ks // 2)
        self.mlp_shared = nn.Sequential(nn.Conv(label_nc, nhidden, ks, padding=pw), nn.ReLU())
        self.mlp_gamma = nn.Conv(nhidden, norm_nc, ks, padding=pw)
        self.mlp_beta = nn.Conv(nhidden, norm_nc, ks, padding=pw)

    def execute(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = nn.interpolate(segmap, size=x.shape[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = ((normalized * (1 + gamma)) + beta)
        return out
