from torch import nn
import torch

class SparseMaxPool(nn.Module):
    def __init__(self, cfg):
        super(SparseMaxPool, self).__init__()
        pooling_counts = cfg.NUM_SCALE_LAYERS  
        N = cfg.NUM_CLIPS
        mask2d = torch.zeros(N, N, dtype=torch.bool) 
        mask2d[range(N), range(N)] = 1  

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c): 
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2
        
        poolers = [nn.MaxPool1d(2,1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3,2)] + [nn.MaxPool1d(2,1) for _ in range(c - 1)]
            )

        # poolers = [nn.AvgPool1d(2,1) for _ in range(pooling_counts[0])]
        # for c in pooling_counts[1:]:
        #     poolers.extend(
        #         [nn.AvgPool1d(3,2)] + [nn.AvgPool1d(2,1) for _ in range(c - 1)]
        #     )

        self.mask2d = mask2d
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N).cuda()
        mask2d = self.mask2d.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1).cuda()
        map2d[:, :, range(N), range(N)] = x
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
        return map2d, mask2d


class SparseBoundaryAdd(nn.Module):
    def __init__(self, cfg):
        super(SparseBoundaryAdd, self).__init__()
        pooling_counts = cfg.NUM_SCALE_LAYERS   
        N = cfg.NUM_CLIPS
        mask2d = torch.zeros(N, N, dtype=torch.bool) 
        mask2d[range(N), range(N)] = 1  

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c): 
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2
        
        self.mask2d = mask2d
        self.maskij = maskij

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N).cuda()
        mask2d = self.mask2d.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1).cuda()
        map2d[:, :, range(N), range(N)] = x
        for (i, j) in self.maskij:
            tmp = x[:, :, i] + x[:, :, j]
            map2d[:, :, i, j] = tmp
        return map2d, mask2d


class SparseBoundaryCat(nn.Module):
    def __init__(self, cfg):
        super(SparseBoundaryCat, self).__init__()
        pooling_counts = cfg.NUM_SCALE_LAYERS   
        N = cfg.NUM_CLIPS
        mask2d = torch.zeros(N, N, dtype=torch.bool) 
        mask2d[range(N), range(N)] = 1  

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c): 
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        self.mask2d = mask2d
        self.maskij = maskij

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, 2 * D, N, N).cuda()
        mask2d = self.mask2d.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1).cuda()
        map2d[:, :, range(N), range(N)] = x.repeat(1, 2, 1)

        for (i, j) in self.maskij:
            tmp = torch.cat((x[:, :, i], x[:, :, j]), dim=1)
            map2d[:, :, i, j] = tmp
        return map2d, mask2d
