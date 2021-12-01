import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import pdb
from torch.autograd import Variable


class DotFuse(nn.Module):

    def __init__(self, cfg):
        super(DotFuse, self).__init__()
        self.cfg = cfg
        self.txt_linear = nn.Linear(cfg.TXT_INPUT_SIZE, cfg.HIDDEN_SIZE)
        self.vis_conv = nn.Conv2d(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)

    def forward(self, choice_map, txt_h):
        txt_pool = torch.max(txt_h, dim=1)[0]  # B, C
        txt_map = self.txt_linear(txt_pool)[:,:,None,None]  # 4, 512, 1, 1
        choice_map = self.vis_conv(choice_map)  # 4, 512, 128, 128
        fused_map = F.normalize(txt_map * choice_map)
        return fused_map


class DynamicFuse(nn.Module):
    def __init__(self, cfg):
        super(DynamicFuse, self).__init__()

        self.cfg = cfg
        self.txt_linear_b1 = nn.Linear(cfg.TXT_INPUT_SIZE, cfg.HIDDEN_SIZE)
        self.vis_conv_b1 = nn.Conv2d(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)
        self.txt_linear_b2_a = nn.Linear(cfg.TXT_INPUT_SIZE, cfg.HIDDEN_SIZE)
        self.vis_conv_b2_a = nn.Conv2d(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)
        self.txt_linear_b2_b = nn.Linear(cfg.TXT_INPUT_SIZE, cfg.HIDDEN_SIZE)
        self.vis_conv_b2_b = nn.Conv2d(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(True)

    def forward(self, choice_map, txt_h):
        txt_pool = torch.max(txt_h, dim=1)[0]  # B, C
        txt_h_b1 = self.txt_linear_b1(txt_pool)[:,:,None,None]  # B, C, 1, 1
        map_h_b1 = self.vis_conv_b1(choice_map)  # B ,C, T, T
        fused_b1 = F.normalize(txt_h_b1 * map_h_b1)
        bsz, dimc, clip_num, _ = map_h_b1.size()  # B, C, T, T

        txt_h_b2_a = self.txt_linear_b2_a(txt_h)  # B, L, C
        map_h_b2_a = self.vis_conv_b2_a(choice_map)  # B, C, T, T
        attn_weight = self.softmax(torch.matmul(txt_h_b2_a, map_h_b2_a.view(bsz, dimc, -1)) / math.sqrt(dimc))  # B, L, T*T
        # attn_weight = self.softmax(torch.matmul(txt_h_b2_a, map_h_b2_a.view(bsz, dimc, -1)))  # B, L, T*T
        txt_h_b2_b = self.txt_linear_b2_b(txt_h)  # B, L, C
        map_h_b2_b = self.vis_conv_b2_b(choice_map)  # B, C, T, T
        txt_attn = torch.matmul(txt_h_b2_b.transpose(-1, -2), attn_weight).view(bsz, -1, clip_num, clip_num)  # B, C, T, T
        fused_b2 = F.normalize(txt_attn * map_h_b2_b)

        fused_h = self.relu(fused_b1 + fused_b2)
        return fused_h


class OnlyDynamic(nn.Module):
    def __init__(self, cfg):
        super(OnlyDynamic, self).__init__()

        self.cfg = cfg
        # self.txt_linear_b1 = nn.Linear(cfg.TXT_INPUT_SIZE, cfg.HIDDEN_SIZE)
        # self.vis_conv_b1 = nn.Conv2d(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)
        self.txt_linear_b2_a = nn.Linear(cfg.TXT_INPUT_SIZE, cfg.HIDDEN_SIZE)
        self.vis_conv_b2_a = nn.Conv2d(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)
        self.txt_linear_b2_b = nn.Linear(cfg.TXT_INPUT_SIZE, cfg.HIDDEN_SIZE)
        self.vis_conv_b2_b = nn.Conv2d(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE, 1, 1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(True)

    def forward(self, choice_map, txt_h):
        # txt_pool = torch.max(txt_h, dim=1)[0]  # B, C
        # txt_h_b1 = self.txt_linear_b1(txt_pool)[:,:,None,None]  # B, C, 1, 1
        # map_h_b1 = self.vis_conv_b1(choice_map)  # B ,C, T, T
        # fused_b1 = F.normalize(txt_h_b1 * map_h_b1)
        bsz, dimc, clip_num, _ = choice_map.size()  # B, C, T, T
        dimc = dimc // 2

        txt_h_b2_a = self.txt_linear_b2_a(txt_h)  # B, L, C
        map_h_b2_a = self.vis_conv_b2_a(choice_map)  # B, C, T, T
        attn_weight = self.softmax(torch.matmul(txt_h_b2_a, map_h_b2_a.view(bsz, dimc, -1)) / math.sqrt(dimc))  # B, L, T*T
        # attn_weight = self.softmax(torch.matmul(txt_h_b2_a, map_h_b2_a.view(bsz, dimc, -1)))  # B, L, T*T
        txt_h_b2_b = self.txt_linear_b2_b(txt_h)  # B, L, C
        map_h_b2_b = self.vis_conv_b2_b(choice_map)  # B, C, T, T
        txt_attn = torch.matmul(txt_h_b2_b.transpose(-1, -2), attn_weight).view(bsz, -1, clip_num, clip_num)  # B, C, T, T
        fused_b2 = F.normalize(txt_attn * map_h_b2_b)

        fused_h = self.relu(fused_b2)
        return fused_h