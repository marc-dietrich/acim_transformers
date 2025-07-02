import torch.nn as nn
from src.layers.encoderLayer import EncoderLayer

class EncoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config['num_layers'])])

    def forward(self, src):
        x = self.embedding(src)
        for layer in self.layers:
            x = layer(x)
        return x