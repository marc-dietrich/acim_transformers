import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from transformers import logging as hf_logging
from datasets import load_dataset
import numpy as np
import evaluate
import warnings

#hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metric = evaluate.load("glue", "cola")
    return metric.compute(predictions=preds, references=labels)

def main():
    # Setup device & model
    model_name = "bert-large-uncased"
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # Load dataset & tokenize
    dataset = load_dataset("glue", "cola")
    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128)
    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Training args with warmup + decay + fp16
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="matthews_correlation",
        greater_is_better=True,
        fp16=True,  # Mixed precision training
        save_total_limit=1,  # keep last 1 checkpoints
        seed=42,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train + evaluate
    trainer.train()

    # Evaluate on validation
    eval_results = trainer.evaluate()
    print(f"Final Validation MCC: {eval_results['eval_matthews_correlation']:.4f}")

if __name__ == "__main__":
    main()
