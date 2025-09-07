import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import NestedTensor


class PositionEmbeddingSineSimple(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        b, c, h, w = x.shape
        y = torch.arange(h, dtype=x.dtype, device=x.device).unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(w, dtype=x.dtype, device=x.device).unsqueeze(0).repeat(h, 1)
        
        if self.normalize:
            eps = 1e-6
            y = y / (h + eps) * self.scale
            x_pos = x_pos / (w + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=x.dtype, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)
        
        pos_x = x_pos[..., None] / dim_t
        pos_y = y[..., None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        
        pos = torch.cat((pos_y, pos_x), dim=2)
        pos = pos.unsqueeze(0).repeat(b, 1, 1, 1).permute(0, 3, 1, 2)
        
        return pos