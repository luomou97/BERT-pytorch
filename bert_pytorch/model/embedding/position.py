import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):
    '''
    here using fiexed positional embedding, \ref Transformer
    '''

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)    # <max_len, 1>
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp() # <d_model//2>

        pe[:, 0::2] = torch.sin(position * div_term)    # <max_len, d_model//2>, matrix * vector
        pe[:, 1::2] = torch.cos(position * div_term)    # <max_len, d_model//2>, matrix * vector

        pe = pe.unsqueeze(0)                # <1, max_len, d_model> to consistent with the number of dimension = 3
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]   # rewrite the embedding look up fuction, get the top seq_len position embedding
