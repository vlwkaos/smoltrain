"""Fine-tune a sequence classification model using HuggingFace Trainer."""
import json
import logging
import warnings
from pathlib import Path

# Suppress noisy HF warnings about classification head init
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*Some weights.*were not initialized.*")
warnings.filterwarnings("ignore", message=".*initializing.*from.*pretrained.*")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset
import evaluate
import numpy as np

from . import config as cfg_mod


def run(cfg):
    ds_path = cfg_mod.dataset_path(cfg.name)
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found at {ds_path}. Run: smoltrain gen {cfg.name}")

    # Load dataset
    records = []
    with open(ds_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError("Dataset is empty")

    label2id = {c: i for i, c in enumerate(cfg.classes)}
    id2label = {i: c for i, c in enumerate(cfg.classes)}

    # Convert labels to ints
    for r in records:
        r["label"] = label2id[r["label"]]

    dataset = Dataset.from_list(records)

    # Train/eval split (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_seq_len,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(cfg.classes),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    out_dir = cfg_mod.model_dir(cfg.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",
    )

    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=preds, references=labels)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save best model + tokenizer
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Final eval
    results = trainer.evaluate()
    acc = results.get("eval_accuracy", None)
    if acc is not None:
        print(f"Final eval accuracy: {acc:.4f}")
    print(f"Model saved to {out_dir}")
