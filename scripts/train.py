#!/usr/bin/env python3
"""Fine-tune DistilBERT for sequence classification using HuggingFace Trainer."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--classes", required=True, help="comma-separated class labels")
    p.add_argument("--goal", default="")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=128)
    return p.parse_args()


def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main():
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",")]
    label2id = {c: i for i, c in enumerate(classes)}
    id2label = {i: c for i, c in enumerate(classes)}

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    import evaluate
    import torch
    from torch.utils.data import Dataset

    raw = load_jsonl(args.dataset)
    # 90/10 split
    n = len(raw)
    split = max(1, int(n * 0.9))
    train_raw, val_raw = raw[:split], raw[split:]

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    class TextDataset(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]
            enc = tokenizer(
                rec["input"],
                truncation=True,
                max_length=args.max_seq_len,
                padding="max_length",
            )
            enc["labels"] = label2id[rec["label"]]
            return {k: torch.tensor(v) for k, v in enc.items()}

    train_ds = TextDataset(train_raw)
    val_ds = TextDataset(val_raw)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id,
    )

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"model saved to {args.output}")


if __name__ == "__main__":
    main()
