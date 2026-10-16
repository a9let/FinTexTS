import torch
from torch import nn

class GRUTextModel(nn.Module):
    def __init__(self, hidden_size, num_layers, input_size = 384):
        super().__init__()
        self.hidden_size = hidden_size
       
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return out[:, -1, :]