#!/usr/bin/env python3
"""
mlx-lm LoRA fine-tuning for smoltrain classifiers.
Called by: smoltrain train <task>

Usage:
  python3 scripts/train.py \
    --model Qwen/Qwen3-0.6B \
    --dataset ~/.local/share/smoltrain/routing/train.jsonl \
    --output ~/.local/share/smoltrain/routing/model \
    --epochs 3 \
    --classes code,edit,chat,research \
    --goal "Classify an LLM request into one of these routing buckets"
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def make_chat_example(goal: str, classes: list[str], input_text: str, label: str) -> dict:
    """Convert a (input, label) pair into a chat fine-tuning example."""
    class_list = "|".join(classes)
    return {
        "messages": [
            {
                "role": "system",
                "content": f"Classify the following request into one of: {class_list}\nRespond with only the class name."
            },
            {
                "role": "user",
                "content": input_text
            },
            {
                "role": "assistant",
                "content": label
            }
        ]
    }


def load_dataset(path: Path, goal: str, classes: list[str]) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            examples.append(make_chat_example(goal, classes, ex["input"], ex["label"]))
    return examples


def split_dataset(examples: list[dict], val_frac: float = 0.1):
    import random
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * val_frac))
    return examples[n_val:], examples[:n_val]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--goal", required=True)
    args = parser.parse_args()

    try:
        import mlx_lm
    except ImportError:
        print("ERROR: mlx-lm not installed. Run: pip install mlx-lm", file=sys.stderr)
        sys.exit(1)

    classes = args.classes.split(",")
    print(f"Classes: {classes}")
    print(f"Loading dataset from {args.dataset}...")

    examples = load_dataset(args.dataset, args.goal, classes)
    train_data, val_data = split_dataset(examples)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # Write to temp JSONL files for mlx-lm
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = Path(tmpdir) / "train.jsonl"
        val_path = Path(tmpdir) / "valid.jsonl"
        test_path = Path(tmpdir) / "test.jsonl"

        for path, data in [(train_path, train_data), (val_path, val_data), (test_path, val_data)]:
            with open(path, "w") as f:
                for ex in data:
                    f.write(json.dumps(ex) + "\n")

        args.output.mkdir(parents=True, exist_ok=True)

        # Build mlx_lm.lora command
        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", args.model,
            "--train",
            "--data", tmpdir,
            "--adapter-path", str(args.output / "adapters"),
            "--num-layers", str(args.lora_rank),
            "--batch-size", str(args.batch_size),
            "--num-iterations", str(len(train_data) * args.epochs // args.batch_size),
            "--learning-rate", str(args.lr),
            "--max-seq-length", str(args.max_seq_len),
            "--steps-per-eval", "50",
            "--save-every", "100",
        ]

        print("Running:", " ".join(cmd))
        import subprocess
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

        # Fuse adapters into full model
        fuse_cmd = [
            sys.executable, "-m", "mlx_lm.fuse",
            "--model", args.model,
            "--adapter-path", str(args.output / "adapters"),
            "--save-path", str(args.output),
        ]
        print("Fusing adapters:", " ".join(fuse_cmd))
        result = subprocess.run(fuse_cmd)
        if result.returncode != 0:
            print("WARNING: adapter fuse failed — keeping unfused adapters")

    print(f"Done. Model at {args.output}")


if __name__ == "__main__":
    main()
