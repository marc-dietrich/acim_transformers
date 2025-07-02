import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from torch.optim import AdamW
import math
import tqdm

# --- Config ---
class Config:
    vocab_size = 50257
    n_embd = 512
    n_layer = 1
    n_head = 8
    max_len = 128
    dropout = 0.1

# --- Minimal GPT2 Decoder Layer ---
class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = nn.MultiheadAttention(config.n_embd, config.n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, attn_mask=None):
        # causal mask must be applied in attn_mask here
        x_ = self.ln1(x)
        attn_output, _ = self.attn(x_, x_, x_, attn_mask=attn_mask)
        x = x + attn_output
        x_ = self.ln2(x)
        x = x + self.mlp(x_)
        return x

# --- GPT2 Model ---
class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.max_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.size()
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        # Create causal mask (lower triangular)
        seq_len = input_ids.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).bool()
        causal_mask = causal_mask == 0  # mask out upper triangle (True means masked)


        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# --- Data Preparation ---
print("Loading dataset and tokenizer...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def encode(examples):
    return tokenizer(examples["text"], truncation=True, max_length=Config.max_len, padding="max_length")

dataset = dataset.map(encode, batched=True)
dataset.set_format(type="torch", columns=["input_ids"])

train_loader = DataLoader(dataset["train"], batch_size=16, shuffle=True)
val_loader = DataLoader(dataset["validation"], batch_size=16)

# --- Training and evaluation ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2Model(Config()).to(device)
optimizer = AdamW(model.parameters(), lr=5e-4)

def evaluate():
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=tokenizer.pad_token_id)
            total_loss += loss.item() * (shift_labels != tokenizer.pad_token_id).sum().item()
            total_tokens += (shift_labels != tokenizer.pad_token_id).sum().item()
    perplexity = math.exp(total_loss / total_tokens)
    print(f"Validation Perplexity: {perplexity:.4f}")
    model.train()
    return perplexity

print("Starting training...")
for epoch in range(3):
    for batch in tqdm.tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=tokenizer.pad_token_id)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} done")
    evaluate()
