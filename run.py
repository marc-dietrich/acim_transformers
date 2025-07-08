import torch
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from transformers import logging as hf_logging
from datasets import load_dataset
import numpy as np
import evaluate
import warnings
import argparse
import time
import torchinfo

#hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

from einops import rearrange
import torch
from torch import nn

def blockdiag_matmul(x, w):
    return torch.einsum(
        "bnm,...bm->...bn", w, x.view(*x.shape[:-1], w.shape[0], w.shape[-1])
    ).reshape(*x.shape)

class MonarchMatrix(nn.Module):
    def __init__(self, L, R, bias=None):
        super().__init__()
        self.sqrt_n = int(L.shape[0])
        #self.L = nn.Parameter(torch.randn((sqrt_n, sqrt_n, sqrt_n)))
        #self.R = nn.Parameter(torch.randn((sqrt_n, sqrt_n, sqrt_n)))
        self.L = nn.Parameter(L)
        self.R = nn.Parameter(R)
        self.bias = bias

    def forward(self, x):
        x = rearrange(x, "... (m n) -> ... (n m)", n=self.sqrt_n)
        x = blockdiag_matmul(x, self.L)
        x = rearrange(x, "... (m n) -> ... (n m)", n=self.sqrt_n)
        x = blockdiag_matmul(x, self.R)
        x = rearrange(x, "... (m n) -> ... (n m)", n=self.sqrt_n)
        if self.bias is not None:
            x = x + self.bias
        return x

def project_to_monarch(A: torch.Tensor):
    """
    Project a square matrix A (shape [n, n], where n = m^2) to a Monarch matrix L @ R,
    returning L, R each as block-diagonal tensors of shape [m, m, m].

    Args:
        A (torch.Tensor): Input square matrix of shape [n, n], where n = m^2.

    Returns:
        L, R (torch.Tensor, torch.Tensor): Monarch-form tensors of shape [m, m, m].
    """
    assert A.ndim == 2
    n = A.size(0)
    m = int(n**0.5)
    assert m * m == n, "Matrix dimension must be a perfect square"

    # Reshape A to a 4D tensor: [m, m, m, m]
    A_reshaped = A.view(m, m, m, m).permute(1, 3, 0, 2).contiguous()
    # Now A_reshaped[j, k] is block M_{jk} ∈ R^{m×m}

    L = torch.empty(m, m, m, dtype=A.dtype, device=A.device)
    R = torch.empty(m, m, m, dtype=A.dtype, device=A.device)

    for j in range(m):
        for k in range(m):
            M_jk = A_reshaped[j, k]  # Shape: [m, m]
            try:
                u, s, vh = torch.linalg.svd(M_jk, full_matrices=False)
            except RuntimeError:
                # Fallback for numerical stability
                u, s, vh = torch.svd(M_jk)

            # Take the leading singular vectors (rank-1 approx)
            u1 = u[:, 0]
            v1 = vh[0, :]

            L[:, j, k] = u1
            R[k, j, :] = v1  # match the indexing from paper

    return L, R


def toMonarch(model):
    replacements = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weights = module.weight.data.clone()
            if weights.shape[0] != weights.shape[1]:
                # pad
                target = max(weights.shape)
                weights = torch.nn.functional.pad(weights, (0, target - weights.shape[1], 0, target - weights.shape[0]), mode='constant', value=0)

            L, R = project_to_monarch(weights)

            bias = module.bias.data.clone() if module.bias is not None else None
            replacements.append((name, MonarchMatrix(L, R, bias)))

    # Apply replacements
    for name, monarch_matrix in replacements:
        parent_module = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, parts[-1], monarch_matrix)

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on CoLA")
    parser.add_argument("--monarch", action="store_true", help="Use Monarch")
    parser.add_argument("--model_name", type=str, default="bert-large-uncased", help="Name of the pretrained model")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of output labels for classification")
    parser.add_argument("--train", action="store_true", help="Whether to train the model")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load a pre-trained Monarch model")

    args = parser.parse_args()
    return args

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metric = evaluate.load("glue", "cola")
    return metric.compute(predictions=preds, references=labels)

def main():
    # Setup device & model
    args = parse_args()


    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=args.num_labels)

    if args.monarch:
        start = time.time()
        model = toMonarch(model)
        print(f"Monarch conversion took {time.time() - start:.2f} seconds")
        torchinfo.summary(model, depth=10)
        quit()

    if args.load_model:
        model = torch.load(args.load_model)

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

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
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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
        save_total_limit=1,  # keep last 2 checkpoints
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
    if args.train:
        trainer.train()
        trainer.save_model()  # Save the final model

    # Evaluate on validation
    eval_results = trainer.evaluate()
    print(f"Final Validation MCC: {eval_results['eval_matthews_correlation']:.4f}")

if __name__ == "__main__":
    main()
