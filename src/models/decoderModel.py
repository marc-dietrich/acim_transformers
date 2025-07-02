import torch
import torch.nn as nn
from src.layers.decoderLayer import DecoderLayer

def generate_causal_mask(seq_len, device):
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask.to(device)

class DecoderModel(nn.Module):
    def __init__(self, config, decoder_only=True):
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.layers = nn.ModuleList([DecoderLayer(config, decoder_only=decoder_only) for _ in range(config['num_layers'])])
        self.mask = generate_causal_mask(config['seq_length'], config['device'])

    def forward(self, tgt, memory):
        x = self.embedding(tgt)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=self.mask)
        return x