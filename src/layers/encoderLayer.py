import torch.nn as nn
import torch.nn.functional as F
import torch

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config['d_model'], config['n_heads'], batch_first=True)
        self.linear1 = nn.Linear(config['d_model'], config['dim_ff'])
        self.dropout = nn.Dropout(config['dropout'])
        self.linear2 = nn.Linear(config['dim_ff'], config['d_model'])
        self.norm1 = nn.LayerNorm(config['d_model'])
        self.norm2 = nn.LayerNorm(config['d_model'])

    def forward(self, src, src_mask=None):
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + attn_output)
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        return self.norm2(src + ff_output)