import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, config, decoder_only=True):
        super().__init__()
        self.decoder_only = decoder_only

        self.self_attn = nn.MultiheadAttention(
            config['d_model'], config['n_heads'], batch_first=True
        )

        if not decoder_only:
            self.cross_attn = nn.MultiheadAttention(
                config['d_model'], config['n_heads'], batch_first=True
            )
            self.norm2 = nn.LayerNorm(config['d_model'])

        self.linear1 = nn.Linear(config['d_model'], config['dim_ff'])
        self.dropout = nn.Dropout(config['dropout'])
        self.linear2 = nn.Linear(config['dim_ff'], config['d_model'])

        self.norm1 = nn.LayerNorm(config['d_model'])
        self.norm3 = nn.LayerNorm(config['d_model'])

    def forward(self, tgt, memory=None, tgt_mask=None, memory_mask=None):
        # Self-attention (masked in decoder)
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + tgt2)

        if not self.decoder_only:
            # Cross-attention (only in encoder-decoder)
            tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
            tgt = self.norm2(tgt + tgt2)

        # Feedforward
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + ff_output)
        return tgt
