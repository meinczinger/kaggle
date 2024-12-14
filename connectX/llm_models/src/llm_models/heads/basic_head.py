import torch
from torch import nn
from torch.nn import functional as F

class Head(nn.Module):
    def __init__(self, embed_size: int, head_size: int, bias: bool, context: int, dropout: float):
        super().__init__()
        self.queries = nn.Linear(embed_size, head_size, bias=bias)
        self.keys = nn.Linear(embed_size, head_size, bias=bias)
        self.values = nn.Linear(embed_size, head_size, bias=bias)

        self.register_buffer('tril', torch.tril(torch.ones(context, context)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        BS, SL, VS = x.shape
        q = self.queries(x) # BS * SL *54
        k = self.keys(x) # BS * SL *54
        v = self.values(x) # BS * SL *54

        attn_weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        attn_weights = attn_weights.masked_fill(self.tril[:SL, :SL] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)

        x = attn_weights @ v  # BS * SL * 54

        return x