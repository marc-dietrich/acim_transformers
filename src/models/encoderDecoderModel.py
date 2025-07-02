import torch.nn as nn
from src.models.encoderModel import EncoderModel
from src.models.decoderModel import DecoderModel

class EncoderDecoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = EncoderModel(config)
        self.decoder = DecoderModel(config, decoder_only=False)
        self.output_proj = nn.Linear(config['d_model'], config['vocab_size'])

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return self.output_proj(output)