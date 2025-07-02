import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig
from datasets import load_dataset
import evaluate
from torch.optim import AdamW
import tqdm

import warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

# --- CONFIG ---
class Config:
    vocab_size = 30522
    hidden_size = 512
    num_hidden_layers = 6
    num_attention_heads = 8
    intermediate_size = 512  # We're using 4x FF layers of same size
    max_position_embeddings = 512
    dropout = 0.1
    num_labels = 2


# --- CUSTOM FFN ---
class CustomFeedForward(nn.Module):
    def __init__(self, hidden_size, num_layers=4, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers += [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)


# --- ENCODER LAYER ---
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.ffn = CustomFeedForward(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, mask=None):
        # mask: (batch_size, seq_len) with 1=keep, 0=pad
        
        # Convert to key_padding_mask: bool, True = mask
        if mask is not None:
            key_padding_mask = (mask == 0)
        else:
            key_padding_mask = None

        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_output)
        ff_output = self.ffn(x)
        x = self.norm2(x + ff_output)
        return x



# --- ENCODER STACK ---
class CustomEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


# --- CLASSIFICATION MODEL ---
class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = CustomEncoder(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        pos_ids = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(pos_ids)
        x = self.dropout(x)
        x = self.encoder(x, mask=attention_mask)
        cls = self.pooler(x[:, 0])  # [CLS] token
        logits = self.classifier(cls)
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits
        return logits


# --- LOAD DATA ---
print("Loading data...")
dataset = load_dataset("glue", "cola")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True)
val_loader = DataLoader(dataset["validation"], batch_size=32)

# --- TRAIN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()
model = CustomBertForSequenceClassification(config).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
metric = evaluate.load("glue", "cola")

for epoch in range(3):
    model.train()
    for batch in tqdm.tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        loss, logits = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()

    # --- EVAL ---
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu().numpy()
            metric.add_batch(predictions=preds, references=labels.cpu().numpy())

    result = metric.compute()
    print(f"Epoch {epoch+1}: Matthews correlation = {result['matthews_correlation']:.4f}")
