import torch
from torch import nn
import numpy as np
import math, copy, time
import torch.nn.functional as F
from torch.autograd import Variable


class LearnPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=64, dropout=0.1):
        super(LearnPositionalEncoding, self).__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

        nn.init.uniform_(self.pos_embed.weight)

        self.dropout = nn.Dropout(p=dropout)


    def forward(self, q):
        bsz_q, d_model, q_frm = q.shape
        assert q_frm == self.pos_embed.weight.shape[0], (q_frm,self.pos_embed.weight.shape)
        q_pos = self.pos_embed.weight.clone()
        q_pos = q_pos.unsqueeze(0)
        q_pos = q_pos.expand(bsz_q, q_frm, d_model).transpose(1,2)
        # q_pos = q_pos.contiguous().view(bsz_q, q_frm, n_head, d_k)
        q = q + q_pos
        return self.dropout(q)


class FrameAvgPool(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size, stride, use_position, num_clips):
        super(FrameAvgPool, self).__init__()
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)

        if use_position:
            self.pos_embed = LearnPositionalEncoding(d_model=hidden_size, max_len=num_clips)
        else:
            self.pos_embed = None

    def forward(self, visual_input):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.avg_pool(vis_h)
        if self.pos_embed:
            vis_h = self.pos_embed(vis_h) 
        return vis_h


# dynamic graph from knn
def knn(x, y=None, k=5):
    if y is None:
        y = x
    inner = -2 * torch.matmul(y.transpose(2, 1), x) 
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    _, idx = pairwise_distance.topk(k=k, dim=-1)  
    return idx


def get_graph_feature(x, prev_x=None, k=5, idx_knn=None):
    batch_size = x.size(0)
    num_points = x.size(2) 
    x = x.view(batch_size, -1, num_points)
    if idx_knn is None:
        idx_knn = knn(x=x, y=prev_x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx_knn.shape[-1]
    idx_base = torch.arange(0, batch_size, device=x.device ).view(-1, 1, 1) * num_points
    idx = (idx_knn + idx_base).view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous() 
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    return feature


class GCNeXtBlock(nn.Module):
    def __init__(self, channel_in, channel_out, k=3, groups=32, width_group=4):
        super(GCNeXtBlock, self).__init__()
        self.k = k
        width = width_group * groups
        self.tconvs = nn.Sequential(
            nn.Conv1d(channel_in, width, kernel_size=1), nn.ReLU(True),
            nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1), nn.ReLU(True),
            nn.Conv1d(width, channel_out, kernel_size=1),
        ) # temporal graph

        self.sconvs = nn.Sequential(
            nn.Conv2d(channel_in * 2, width, kernel_size=1), nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=(1,self.k), groups=groups,  padding=(0,(self.k-1)//2)), nn.ReLU(True),
            nn.Conv2d(width, channel_out, kernel_size=1),
        ) # semantic graph

        self.relu = nn.ReLU(True)

    def forward(self, x):
        identity = x  # residual
        tout = self.tconvs(x)  # conv on temporal graph

        x_f = get_graph_feature(x, k=self.k) 
        sout = self.sconvs(x_f)  # conv on semantic graph
        sout = sout.max(dim=-1, keepdim=False)[0] 

        out = tout + 2 * identity + sout  
        return self.relu(out)


class GCNeXtMoudle(nn.Module):
    def __init__(self, channel_in, channel_out, k_num, groups, width_group):
        super(GCNeXtMoudle, self).__init__()

        self.backbone = nn.Sequential(
            GCNeXtBlock(channel_in, channel_out, k_num, groups, width_group),
        )

    def forward(self, x):
        gcnext_feature = self.backbone(x)
        return gcnext_feature


class FeatureEncoder(nn.Module):

    def __init__(self, cfg):
        super(FeatureEncoder, self).__init__()
        self.frame_encoder = FrameAvgPool(cfg.FRAME.INPUT_SIZE, cfg.FRAME.HIDDEN_SIZE,cfg.FRAME.KERNEL_SIZE,cfg.FRAME.STRIDE,
                                        cfg.FRAME.USE_POSITION,cfg.FRAME.NUM_CLIPS)
        self.gcnext_layer = GCNeXtMoudle(cfg.GCNEXT.INPUT_SIZE, cfg.GCNEXT.OUTPUT_SIZE, cfg.GCNEXT.K_NUM, cfg.GCNEXT.GROUP_NUM, cfg.GCNEXT.WIDTH_GROUP)
        self.lstm_encoder = nn.LSTM(cfg.LSTM.TXT_INPUT_SIZE, cfg.LSTM.TXT_HIDDEN_SIZE//2 if cfg.LSTM.BIDIRECTIONAL else cfg.LSTM.TXT_HIDDEN_SIZE,
                                       num_layers=cfg.LSTM.NUM_LAYERS, bidirectional=cfg.LSTM.BIDIRECTIONAL, batch_first=True)


    def forward(self, visual_input, textual_input, textual_mask):
        visual_input = visual_input.transpose(1, 2)  
        vis_frame = self.frame_encoder(visual_input)  # B, C, T
        vis_out = self.gcnext_layer(vis_frame)  # B, C, T 
        self.lstm_encoder.flatten_parameters()
        txt_out = self.lstm_encoder(textual_input)[0] * textual_mask  # B, L, C
        return vis_out, txt_out

