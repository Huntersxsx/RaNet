  
import torch
import torch.nn as nn

import torch.autograd as autograd
import torch.cuda.comm as comm
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load
import os, time

curr_dir = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(curr_dir, "src")
_build_path = os.path.join(curr_dir, "build")
os.makedirs(_build_path, exist_ok=True)
graph_attn = load(name="rcca",
            extra_cflags=["-O3"],
            build_directory=_build_path,
            verbose=True,
            sources = [os.path.join(_src_path, f) for f in [
                "lib_cffi.cpp", "ca.cu"
                ]],
            extra_cuda_cflags=["--expt-extended-lambda"])

def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")


class CA_Weight(autograd.Function):
    @staticmethod
    def forward(ctx, t, f):
        # Save context
        n, c, h, w = t.size()
        size = (n, h+w-1, h, w)
        weight = torch.zeros(size, dtype=t.dtype, layout=t.layout, device=t.device)

        graph_attn.ca_forward_cuda(t, f, weight)
        
        # Output
        ctx.save_for_backward(t, f)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors

        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)

        graph_attn.ca_backward_cuda(dw.contiguous(), t, f, dt, df)

        _check_contiguous(dt, df)

        return dt, df

class CA_Map(autograd.Function):
    @staticmethod
    def forward(ctx, weight, g):
        # Save context
        out = torch.zeros_like(g)
        graph_attn.ca_map_forward_cuda(weight, g, out)
        
        # Output
        ctx.save_for_backward(weight, g)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors

        dw = torch.zeros_like(weight)
        dg = torch.zeros_like(g)

        graph_attn.ca_map_backward_cuda(dout.contiguous(), weight, g, dw, dg)

        _check_contiguous(dw, dg)

        return dw, dg

ca_weight = CA_Weight.apply
ca_map = CA_Map.apply


class GraphAttention(nn.Module):
    def __init__(self, feature_size):
        super(GraphAttention,self).__init__()
        self.query_conv = nn.Conv2d(feature_size, feature_size//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(feature_size, feature_size//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(feature_size , feature_size , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_query = self.query_conv(x)
        x_key = self.key_conv(x)
        x_value = self.value_conv(x)

        energy = ca_weight(x_query, x_key)
        attn = F.softmax(energy, 1)
        out = ca_map(attn, x_value)
        out = self.gamma * out + x
        return out



__all__ = ["GraphAttention", "ca_weight", "ca_map"]