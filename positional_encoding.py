import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors              
        mask = tensor_list.mask             
        assert mask is not None
        not_mask = ~mask

        y_embed = not_mask.cumsum(dim=1, dtype=x.dtype)
        x_embed = not_mask.cumsum(dim=2, dtype=x.dtype)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=x.dtype, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2).to(x.dtype) / float(self.num_pos_feats))

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats=256, max_len=50):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.row_embed = nn.Embedding(max_len, num_pos_feats)
        self.col_embed = nn.Embedding(max_len, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        b, c, h, w = x.shape

        row_weight = self.row_embed.weight
        col_weight = self.col_embed.weight
        M_row = row_weight.shape[0]
        M_col = col_weight.shape[0]

        if h <= M_row:
            y_emb = row_weight[:h, :]
        else:
            y_emb = F.interpolate(row_weight.t().unsqueeze(0), size=h, mode='linear', align_corners=False)
            y_emb = y_emb.squeeze(0).t()

        if w <= M_col:
            x_emb = col_weight[:w, :]
        else:
            x_emb = F.interpolate(col_weight.t().unsqueeze(0), size=w, mode='linear', align_corners=False)
            x_emb = x_emb.squeeze(0).t()

        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps, max_len=getattr(args, "max_pos_embed", 50))
    else:
        raise ValueError(f"not supported {args.position_embedding}")
    return position_embedding
