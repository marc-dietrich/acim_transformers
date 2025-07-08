import torch.nn as nn
from src.layers.encoderLayer import EncoderLayer
import torch

class BertEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.position_embeddings = nn.Embedding(config['seq_length'], config['d_model'])
        self.token_type_embeddings = nn.Embedding(config['type_vocab_size'], config['d_model'])
        self.LayerNorm = nn.LayerNorm(config['d_model'], eps=1e-12)
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # [B, seq_len]

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([EncoderLayer(config) for _ in range(config['num_layers'])])

    def forward(self, x):
        # Pass through transformer encoder layers
        for layer in self.layers:
            x = layer(x)

class EncoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbedding(config)  # Assuming pretrained_bert is passed or loaded elsewhere
        self.encoder = Encoder(config)
        
        self.pooler = nn.Sequential(
            nn.Linear(config['d_model'], config['pooler']['hidden_size']),
            nn.Tanh()
        )

        self.classifier = nn.Linear(config['d_model'], config['num_labels'])
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Embed input
        x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)

        x = self.encoder(x)

        # Get [CLS]-like summary vector
        x = x[:, 0, :]
        x = self.pooler(x)

        # Classification head
        logits = self.classifier(x)  # (batch_size, num_labels)

        """
        print(f"Logits shape: {logits.shape}")
        print(f"Labels shape: {labels.shape if labels is not None else 'None'}")
        quit()
        """

        # If labels provided, compute loss
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        else:
            return logits
