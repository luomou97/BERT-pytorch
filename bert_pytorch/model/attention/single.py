import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))    # matmul(<batch_size, len_(Q/K), d_(Q/V)>, <batch_size, d_(Q/V), len_(Q/K)>) => <batch_size, len_(Q/K), len_(Q/K)>

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)    # replace padding position(only using the actual sequence) with -∞, then softmax value similar to 0 \ref Transformer

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
