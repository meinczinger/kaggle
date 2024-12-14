from torch import nn

class ForwadLayer(nn.Module):
    def __init__(self, embed_size: int, bias: bool, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_size, 6 * embed_size, bias=bias),
            nn.GELU(),
            nn.Linear(6 * embed_size, embed_size, bias=bias),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.network(x)
        return x