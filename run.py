import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
from datasets import load_dataset
import evaluate
import tqdm
import os
import warnings

warnings.filterwarnings("ignore")  # suppress warnings, incl. beta/gamma rename

def compute_metrics():
    metric = evaluate.load("glue", "cola")
    def compute(p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        return metric.compute(predictions=preds, references=labels)
    return compute

def tokenize(batch, tokenizer):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128)

def train(model, optimizer, train_loader, val_loader, device):
    metric = compute_metrics()
    os.makedirs("checkpoints", exist_ok=True)

    model.to(device)
    for epoch in range(3):
        model.train()
        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        result = evaluate.load("glue", "cola").compute(predictions=all_preds, references=all_labels)
        print(f"Epoch {epoch+1} Validation Matthews Corr: {result['matthews_correlation']:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/bert_epoch{epoch+1}.pt")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained model + tokenizer
    model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    # example for replacement:
    """old = model.classifier
    new_layer = MyCustomLinear(old.in_features, old.out_features)
    new_layer.linear.weight.data.copy_(old.weight.data)
    new_layer.linear.bias.data.copy_(old.bias.data)
    model.classifier = new_layer"""


    # Load dataset
    dataset = load_dataset("glue", "cola")
    dataset = dataset.map(lambda x: tokenize(x, tokenizer), batched=True)
    dataset.set_format(type='torch', columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(dataset["train"], batch_size=1, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=1)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Train
    train(model, optimizer, train_loader, val_loader, device)

if __name__ == "__main__":
    main()
