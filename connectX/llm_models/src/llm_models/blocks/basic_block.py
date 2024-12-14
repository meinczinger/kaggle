import torch.nn as nn
from llm_models.heads.basic_multi_head import Multihead
from llm_models.layers.basic_forward_layer import ForwadLayer


class Block(nn.Module):
    def __init__(
        self, embed_size: int, n_heads: int, context: int, dropout: int, bias: bool
    ):
        super().__init__()
        head_size = embed_size // n_heads
        self._ma = Multihead(n_heads, head_size, embed_size, context, bias, dropout)
        self.feed_forward = ForwadLayer(embed_size, bias, dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        ln1 = self.ln1(x)
        x = x + self._ma(ln1)
        x = x + self.feed_forward(self.ln2(x))
        return x
