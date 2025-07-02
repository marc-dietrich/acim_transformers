from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np

# 1. Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

# 2. Load GLUE CoLA dataset
dataset = load_dataset("glue", "cola")

# 3. Tokenize the data
def tokenize(batch):
    return tokenizer(batch["sentence"], padding="max_length", truncation=True)

encoded = dataset.map(tokenize, batched=True)
encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 4. Define compute_metrics
metric = evaluate.load("glue", "cola")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)

# 5. Training arguments
args = TrainingArguments(
    "bert-cola",
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_steps=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    load_best_model_at_end=False,
    report_to="none",
)

# 6. Trainer setup
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["validation"],
    compute_metrics=compute_metrics,
)

# 7. Train and evaluate
trainer.train()
results = trainer.evaluate()
print("GLUE CoLA Eval Results:", results)
