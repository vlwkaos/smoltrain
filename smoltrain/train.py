"""Train CharCNN on dataset.jsonl -> models/charcnn_trained.pt + .onnx

Usage:
    python -m smoltrain.train
    python -m smoltrain.train [--data data/dataset.jsonl] [--taxonomy taxonomy.yaml] [--epochs 10] [--seed 42]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from smoltrain.model import CharCNN, encode_text


def load_dataset(data_path: str) -> list[dict]:
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_classes(taxonomy_path: str) -> list[str]:
    with open(taxonomy_path, encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)
    return list(taxonomy["classes"].keys())


def stratified_split(records: list[dict], val_ratio: float = 0.2, seed: int = 42) -> tuple[list, list]:
    rng = random.Random(seed)
    by_label: dict[str, list] = {}
    for r in records:
        by_label.setdefault(r["label"], []).append(r)
    train, val = [], []
    for items in by_label.values():
        items = items[:]
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], class_to_idx: dict[str, int]):
        self.records = records
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        x = torch.tensor(encode_text(r["text"]), dtype=torch.long)
        y = self.class_to_idx[r["label"]]
        return x, y


def compute_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(dim=1) == y).sum().item()
            total += len(y)
    return correct / total if total > 0 else 0.0


def train(
    data_path: str = "data/dataset.jsonl",
    taxonomy_path: str = "taxonomy.yaml",
    epochs: int = 10,
    seed: int = 42,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 3,
    out_dir: str = "models",
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    classes = load_classes(taxonomy_path)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes}")

    records = load_dataset(data_path)
    print(f"Loaded {len(records)} records")

    train_records, val_records = stratified_split(records, val_ratio=0.2, seed=seed)
    print(f"Split: {len(train_records)} train / {len(val_records)} val")

    train_loader = torch.utils.data.DataLoader(
        TextDataset(train_records, class_to_idx), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        TextDataset(val_records, class_to_idx), batch_size=batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharCNN(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pt_path = out_path / "charcnn_trained.pt"
    onnx_path = out_path / "charcnn_trained.onnx"

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)

        avg_loss = total_loss / len(train_records)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * len(y)
        val_loss /= len(val_records)
        val_acc = compute_accuracy(model, val_loader, device)

        print(f"Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "classes": classes}, pt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop (no improvement for {patience} epochs)")
                break

    print(f"\nBest checkpoint -> {pt_path}")

    # reload best for final metrics + export
    ckpt = torch.load(pt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    train_acc = compute_accuracy(model, train_loader, device)
    val_acc = compute_accuracy(model, val_loader, device)
    print(f"Final  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    # ONNX export using encode_text for dummy input
    model.eval()
    dummy = torch.tensor(encode_text("dummy"), dtype=torch.long).unsqueeze(0).to(device)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )
    print(f"ONNX -> {onnx_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.jsonl")
    parser.add_argument("--taxonomy", default="taxonomy.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(data_path=args.data, taxonomy_path=args.taxonomy, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
