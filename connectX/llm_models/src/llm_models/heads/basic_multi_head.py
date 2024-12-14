import torch
from torch import nn
from llm_models.heads.basic_head import Head

class Multihead(nn.Module):
    def __init__(self, n_heads: int, head_size: int, embed_size: int, context: int, bias: bool, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([Head(embed_size, head_size, bias, context, dropout) for _ in range(n_heads)])
        self.combine = nn.Linear(head_size * n_heads, embed_size, bias=bias) # 378 -> 384 (embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        # Each head outputs (BS, SL, head_size)
        x = self.combine(x) # (BS, SL, embed_size)
        x = self.dropout(x)
        return x