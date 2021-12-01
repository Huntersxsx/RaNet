import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sys, os

from .graph_attention import GraphAttention
from models.relation_constructor import get_padded_mask_and_weight

class MapConv(nn.Module):

    def __init__(self, cfg):
        super(MapConv, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_sizes = cfg.HIDDEN_SIZES
        kernel_sizes = cfg.KERNEL_SIZES
        strides = cfg.STRIDES
        paddings = cfg.PADDINGS
        dilations = cfg.DILATIONS
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size]+hidden_sizes
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], k, s, p, d))

    def forward(self, x, mask):
        padded_mask = mask
        for i, pred in enumerate(self.convs):
            x = F.relu(pred(x))
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
            x = x * masked_weight
        return x


class GATBlock(nn.Module):
    def __init__(self, cfg):
        super(GATBlock, self).__init__()
        input_size = cfg.INPUT_SIZE
        inter_size = input_size // cfg.INTER_PROP    
        output_size = cfg.OUTPUT_SIZE
        self.loop_num = cfg.LOOP_NUM
        self.conva = nn.Sequential(nn.Conv2d(input_size, inter_size, 3, padding=1, bias=False),
                                   nn.ReLU(inplace=True),)
        self.gat = GraphAttention(inter_size)
        self.convb = nn.Sequential(nn.Conv2d(inter_size, inter_size, 3, padding=1, bias=False),
                                   nn.ReLU(inplace=True))

        self.convc = nn.Sequential(nn.Conv2d(input_size + inter_size, output_size, 3, padding=1, bias=False),
                                   nn.ReLU(inplace=True))

    def forward(self, x, map_mask):
        map_mask = map_mask.float()
        output = self.conva(x)
        for loop in range(self.loop_num):
            output = self.gat(output)
        output = self.convb(output)
        output = self.convc(torch.cat([x, output], 1)) * map_mask
        return output


class GATModule(nn.Module):
    def __init__(self, cfg):
        super(GATModule, self).__init__()
        self.block_num = cfg.BLOCK_NUM
        self.gat_block = nn.ModuleList([GATBlock(cfg) for _ in range(self.block_num)])

    def forward(self, x, map_mask):
        for i in range(self.block_num):
            x = self.gat_block[i](x, map_mask)
        return x
