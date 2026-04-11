"""Full training pipeline: merge -> train -> eval -> ONNX export -> latency check.

Usage:
    python -m smoltrain.pipeline
    python -m smoltrain.pipeline --data data/dataset.jsonl,data/naturalized.jsonl --epochs 15
"""
import argparse
import json
import logging
import random
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml

from smoltrain.model import CharCNN, encode_text
from smoltrain.train import TextDataset, load_classes, stratified_split


def _step(n: int, name: str) -> None:
    print(f"\n=== Step {n}: {name} ===")


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="smoltrain full pipeline")
    parser.add_argument("--data", default="data/dataset.jsonl,data/naturalized.jsonl",
                        help="Comma-separated JSONL files to merge for training")
    parser.add_argument("--eval", default="data/eval_full.jsonl")
    parser.add_argument("--taxonomy", default="taxonomy.yaml")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args(argv)

    # -------------------------------------------------------------------------
    _step(1, "Setup")
    taxonomy_path = Path(args.taxonomy)
    eval_path = Path(args.eval)
    data_files = [Path(p.strip()) for p in args.data.split(",")]

    if not taxonomy_path.exists():
        print(f"ERROR: taxonomy not found: {taxonomy_path}")
        sys.exit(1)
    print(f"Taxonomy:  {taxonomy_path}")

    for f in data_files:
        if not f.exists():
            print(f"ERROR: data file not found: {f}")
            sys.exit(1)
        print(f"Data:      {f}")

    if not eval_path.exists():
        print(f"ERROR: eval file not found: {eval_path}")
        sys.exit(1)
    print(f"Eval:      {eval_path}")

    classes = load_classes(str(taxonomy_path))
    with open(taxonomy_path, encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)
    taxonomy_cfg = taxonomy.get("config", {})
    f1_floor = taxonomy_cfg.get("f1_floor", 0.85)
    agentic_recall_floor = taxonomy_cfg.get("agentic_recall_floor", 0.87)
    print(f"Classes:   {classes}")
    print(f"f1_floor={f1_floor}  agentic_recall_floor={agentic_recall_floor}")

    # -------------------------------------------------------------------------
    _step(2, "Merge")
    all_records: list[dict] = []
    for f in data_files:
        recs = _load_jsonl(str(f))
        print(f"  {f}: {len(recs)} records")
        all_records.extend(recs)
    dist = Counter(r["label"] for r in all_records)
    print(f"Total: {len(all_records)}  distribution: {dict(dist)}")

    # -------------------------------------------------------------------------
    _step(3, "Train")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    train_records, val_records = stratified_split(all_records, val_ratio=0.2, seed=args.seed)
    print(f"Split: {len(train_records)} train / {len(val_records)} val")

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        TextDataset(train_records, class_to_idx), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        TextDataset(val_records, class_to_idx), batch_size=batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharCNN(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    pt_path = out_path / "charcnn_trained.pt"
    onnx_path = out_path / "charcnn_trained.onnx"

    best_val_loss = float("inf")
    no_improve = 0
    patience = 3
    epochs = args.epochs
    early_stopped = False

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
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(y)
                correct += (logits.argmax(dim=1) == y).sum().item()
                total += len(y)
        val_loss /= len(val_records)
        val_acc = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}"
            f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({"model_state": model.state_dict(), "classes": classes}, pt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch} (no improvement for {patience} epochs)")
                early_stopped = True
                break

    print(f"Best checkpoint -> {pt_path}")
    ckpt = torch.load(pt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])

    # -------------------------------------------------------------------------
    _step(4, "Eval")
    eval_records = _load_jsonl(str(eval_path))
    print(f"Eval set: {len(eval_records)} records from {eval_path}")

    model.eval()
    tp: dict[str, int] = {c: 0 for c in classes}
    fp: dict[str, int] = {c: 0 for c in classes}
    fn: dict[str, int] = {c: 0 for c in classes}
    class_total: Counter = Counter()

    with torch.no_grad():
        for i in range(0, len(eval_records), batch_size):
            batch = eval_records[i : i + batch_size]
            xs = torch.tensor(
                np.stack([encode_text(r["text"]) for r in batch]), dtype=torch.long
            ).to(device)
            preds = model(xs).argmax(dim=1).cpu().tolist()
            for r, p in zip(batch, preds):
                gt = class_to_idx[r["label"]]
                class_total[r["label"]] += 1
                if p == gt:
                    tp[r["label"]] += 1
                else:
                    fp[idx_to_class[p]] += 1
                    fn[r["label"]] += 1

    f1_per_class: dict[str, float] = {}
    recall_per_class: dict[str, float] = {}
    print(f"  {'Class':<14} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    for cls in classes:
        n = class_total[cls]
        acc = tp[cls] / n if n > 0 else 0.0
        prec = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0.0
        rec = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_class[cls] = f1
        recall_per_class[cls] = rec
        flag = " <" if f1 < f1_floor else ""
        print(f"  {cls:<14} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {n:>8}{flag}")

    overall_acc = sum(tp.values()) / len(eval_records)
    print(f"  Overall accuracy: {overall_acc:.4f}")

    f1_pass = all(f1_per_class[c] >= f1_floor for c in classes)
    agentic_recall = recall_per_class.get("agentic", float("nan"))
    agentic_recall_pass = agentic_recall >= agentic_recall_floor if not np.isnan(agentic_recall) else True

    # -------------------------------------------------------------------------
    _step(5, "ONNX Export")
    model.eval()
    dummy = torch.tensor(encode_text("dummy"), dtype=torch.long).unsqueeze(0).to(device)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    for _logger_name in ("torch.onnx", "torch.onnx._internal", "torch.onnx._internal.exporter"):
        logging.getLogger(_logger_name).setLevel(logging.ERROR)
    # torch.onnx writes progress to stdout — redirect to devnull during export
    import os as _os
    _devnull = open(_os.devnull, "w")
    _old_stdout, _old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _devnull, _devnull
    try:
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=18,
        )
    finally:
        sys.stdout, sys.stderr = _old_stdout, _old_stderr
        _devnull.close()
    onnx_size_kb = onnx_path.stat().st_size / 1024
    print(f"ONNX -> {onnx_path}  ({onnx_size_kb:.1f} KB)")
    onnx_data_path = Path(str(onnx_path) + ".data")
    if onnx_data_path.exists():
        data_size_kb = onnx_data_path.stat().st_size / 1024
        print(f"  sidecar -> {onnx_data_path}  ({data_size_kb:.1f} KB)")

    # -------------------------------------------------------------------------
    _step(6, "Latency Check")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample_arr = encode_text(eval_records[0]["text"]).reshape(1, -1).astype(np.int64)
    for _ in range(20):
        session.run(["logits"], {"input_ids": sample_arr})

    n_runs = 200
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(["logits"], {"input_ids": sample_arr})
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    p50_ms = times[n_runs // 2]
    p95_ms = times[int(n_runs * 0.95)]
    latency_pass = p50_ms <= 50.0
    print(f"n={n_runs}  p50={p50_ms:.2f}ms  p95={p95_ms:.2f}ms  {'PASS' if latency_pass else 'FAIL'}")

    # -------------------------------------------------------------------------
    print("\n" + "=" * 54)
    print("PIPELINE SUMMARY")
    print("=" * 54)
    print(f"  Train records:    {len(all_records)}")
    print(f"  Eval records:     {len(eval_records)}")
    print(f"  Overall acc:      {overall_acc:.4f}")
    for cls in classes:
        flag = "PASS" if f1_per_class[cls] >= f1_floor else "FAIL"
        print(f"  F1[{cls}]:{'':<{12 - len(cls)}} {f1_per_class[cls]:.4f}  [{flag}]")
    if "agentic" in classes:
        flag = "PASS" if agentic_recall_pass else "FAIL"
        print(f"  Agentic recall:   {agentic_recall:.4f}  [{flag}]")
    print(f"  F1 floor ({f1_floor}):   {'PASS' if f1_pass else 'FAIL'}")
    print(f"  Latency p50:      {p50_ms:.2f}ms  [{'PASS' if latency_pass else 'FAIL'}]")
    print(f"  Latency p95:      {p95_ms:.2f}ms")
    print(f"  ONNX:             {onnx_path}  ({onnx_size_kb:.1f} KB)")
    print("=" * 54)
    overall_pass = f1_pass and latency_pass and agentic_recall_pass
    print(f"  Overall: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 54)


if __name__ == "__main__":
    main()
