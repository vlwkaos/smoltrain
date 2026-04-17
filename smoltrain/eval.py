"""6-axis evaluation harness for CharCNN ONNX model.

Usage:
    python -m smoltrain.eval
    python -m smoltrain.eval [--model models/charcnn_trained.onnx] [--data data/dataset.jsonl]
                             [--taxonomy taxonomy.yaml] [--world data/world.json]
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from smoltrain.model import encode_text
from smoltrain.train import load_classes, load_dataset, stratified_split


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def build_session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def predict_batch(session: ort.InferenceSession, texts: list[str]) -> list[int]:
    arr = np.stack([encode_text(t) for t in texts]).astype(np.int64)
    logits = session.run(["logits"], {"input_ids": arr})[0]
    return logits.argmax(axis=1).tolist()


def predict_one(session: ort.InferenceSession, text: str) -> int:
    arr = encode_text(text).reshape(1, -1).astype(np.int64)
    logits = session.run(["logits"], {"input_ids": arr})[0]
    return int(logits.argmax(axis=1)[0])


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def per_class_f1(records: list[dict], preds: list[int], classes: list[str]) -> dict:
    """Returns {class: {precision, recall, f1, support}}."""
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    tp = [0] * n
    fp = [0] * n
    fn = [0] * n
    for r, p in zip(records, preds):
        t = idx[r["label"]]
        if p == t:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    result = {}
    for i, cls in enumerate(classes):
        prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        rec = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result[cls] = {"precision": prec, "recall": rec, "f1": f1, "support": tp[i] + fn[i]}
    return result


def subset_accuracy(records: list[dict], preds: list[int], classes: list[str],
                    filter_fn) -> float:
    idx = {c: i for i, c in enumerate(classes)}
    filtered = [(r, p) for r, p in zip(records, preds) if filter_fn(r)]
    if not filtered:
        return float("nan")
    correct = sum(1 for r, p in filtered if idx[r["label"]] == p)
    return correct / len(filtered)


# ---------------------------------------------------------------------------
# Axis evaluations
# ---------------------------------------------------------------------------

def axis1_per_class_f1(records, preds, classes, taxonomy_cfg):
    f1_floor = taxonomy_cfg.get("f1_floor", 0.85)
    agentic_recall_floor = taxonomy_cfg.get("agentic_recall_floor", 0.87)

    stats = per_class_f1(records, preds, classes)

    print("\n--- Axis 1: Per-class F1 ---")
    print(f"{'Class':<14} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    for cls in classes:
        s = stats[cls]
        print(f"{cls:<14} {s['precision']:>10.4f} {s['recall']:>8.4f} {s['f1']:>8.4f} {s['support']:>8}")

    pass_f1 = all(s["f1"] >= f1_floor for s in stats.values())
    warn_agentic = "agentic" in stats and stats["agentic"]["recall"] < agentic_recall_floor

    verdict = "PASS" if pass_f1 else "FAIL"
    if warn_agentic:
        print(f"  WARN: agentic recall {stats['agentic']['recall']:.4f} < {agentic_recall_floor}")
    print(f"  Axis 1: {verdict} (all F1 >= {f1_floor})")
    return verdict == "PASS", stats


def axis2_crosslingual(records, preds, classes, taxonomy_cfg):
    gap_threshold = 10.0  # pp
    idx = {c: i for i, c in enumerate(classes)}

    print("\n--- Axis 2: Cross-lingual consistency ---")
    languages = sorted({r["lang"] for r in records})
    # per class per lang accuracy
    cell: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r, p in zip(records, preds):
        cell[r["label"]][r["lang"]].append(int(idx[r["label"]] == p))

    flagged = False
    header = f"{'Class':<14}" + "".join(f"{lang:>10}" for lang in languages)
    print(header)
    for cls in classes:
        row = f"{cls:<14}"
        accs = {}
        for lang in languages:
            items = cell[cls].get(lang, [])
            acc = sum(items) / len(items) if items else float("nan")
            accs[lang] = acc
            row += f"{acc:>10.4f}" if not np.isnan(acc) else f"{'N/A':>10}"
        valid = [v for v in accs.values() if not np.isnan(v)]
        if len(valid) >= 2:
            gap = (max(valid) - min(valid)) * 100
            if gap > gap_threshold:
                row += f"  <- gap {gap:.1f}pp"
                flagged = True
        print(row)

    verdict = "FAIL" if flagged else "PASS"
    print(f"  Axis 2: {verdict} (no language gap > {gap_threshold}pp)")
    return verdict == "PASS"


def axis3_codeswitching(records, preds, classes):
    idx = {c: i for i, c in enumerate(classes)}

    print("\n--- Axis 3: Code-switching robustness ---")
    overall_acc = sum(1 for r, p in zip(records, preds) if idx[r["label"]] == p) / len(records)
    mixed = [(r, p) for r, p in zip(records, preds) if r["lang"] == "mixed"]
    mixed_acc = (sum(1 for r, p in mixed if idx[r["label"]] == p) / len(mixed)
                 if mixed else float("nan"))
    print(f"  Overall accuracy: {overall_acc:.4f}")
    print(f"  Mixed accuracy:   {mixed_acc:.4f}  (n={len(mixed)})")

    if np.isnan(mixed_acc):
        print("  Axis 3: SKIP (no mixed samples)")
        return True  # ^ no mixed data: skip, don't penalize

    gap = (overall_acc - mixed_acc) * 100
    verdict = "PASS" if gap <= 5.0 else "FAIL"
    print(f"  Gap: {gap:.1f}pp  Axis 3: {verdict}")
    return verdict == "PASS"


def axis4_hard_negatives(records, preds, classes, world: dict):
    """
    Hard negative: label=A but text contains key_signals from B's discriminator vs A.
    class_B.cross_class_discriminators.vs_A.key_signals -> these identify B vs A.
    A sample labeled A that matches B's signals is confusable.
    """
    idx = {c: i for i, c in enumerate(classes)}
    cls_data = world.get("classes", {})

    print("\n--- Axis 4: Hard negative accuracy ---")

    results = []
    for cls_a in classes:
        for cls_b in classes:
            if cls_a == cls_b:
                continue
            vs_key = f"vs_{cls_a}"
            disc = cls_data.get(cls_b, {}).get("cross_class_discriminators", {}).get(vs_key, {})
            signals = [s.lower() for s in disc.get("key_signals", [])
                       if not s.startswith("no ") and len(s) >= 3]
            if not signals:
                continue

            hard = [
                (r, p) for r, p in zip(records, preds)
                if r["label"] == cls_a and any(sig in r["text"].lower() for sig in signals)
            ]
            if not hard:
                continue

            acc = sum(1 for r, p in hard if idx[r["label"]] == p) / len(hard)
            results.append((cls_a, cls_b, acc, len(hard)))
            print(f"  {cls_a} vs {cls_b}: acc={acc:.4f}  n={len(hard)}")

    if not results:
        print("  No hard negative pairs found")
        return True

    all_pass = all(acc >= 0.75 for _, _, acc, _ in results)
    verdict = "PASS" if all_pass else "FAIL"
    print(f"  Axis 4: {verdict} (all hard-negative acc >= 0.75)")
    return verdict == "PASS"


def axis5_length_buckets(records, preds, classes):
    idx = {c: i for i, c in enumerate(classes)}

    print("\n--- Axis 5: Length buckets ---")
    buckets = {"short (<=30)": [], "medium (31-120)": [], "long (>120)": []}

    for r, p in zip(records, preds):
        n = len(r["text"])
        correct = int(idx[r["label"]] == p)
        if n <= 30:
            buckets["short (<=30)"].append(correct)
        elif n <= 120:
            buckets["medium (31-120)"].append(correct)
        else:
            buckets["long (>120)"].append(correct)

    all_pass = True
    for name, items in buckets.items():
        if items:
            acc = sum(items) / len(items)
            ok = acc >= 0.70
            all_pass = all_pass and ok
            print(f"  {name:<18}: acc={acc:.4f}  n={len(items)}  {'OK' if ok else 'FAIL'}")
        else:
            print(f"  {name:<18}: no samples")

    verdict = "PASS" if all_pass else "FAIL"
    print(f"  Axis 5: {verdict} (all buckets >= 0.70)")
    return verdict == "PASS"


def axis6_latency(session: ort.InferenceSession, sample_text: str, n: int = 1000):
    print("\n--- Axis 6: Latency ---")
    arr = encode_text(sample_text).reshape(1, -1).astype(np.int64)
    # warmup
    for _ in range(10):
        session.run(["logits"], {"input_ids": arr})

    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        session.run(["logits"], {"input_ids": arr})
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    median_ms = times[len(times) // 2]
    p95_ms = times[int(len(times) * 0.95)]

    print(f"  n={n}  median={median_ms:.2f}ms  p95={p95_ms:.2f}ms")
    verdict = "PASS" if median_ms <= 50.0 else "FAIL"
    print(f"  Axis 6: {verdict} (median <= 50ms)")
    return verdict == "PASS", median_ms, p95_ms


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------

def evaluate(
    model_path: str = "models/charcnn_trained.onnx",
    data_path: str = "data/dataset.jsonl",
    taxonomy_path: str = "taxonomy.yaml",
    world_path: str = "data/world.json",
    eval_data_path: str | None = None,
    seed: int = 42,
    batch_size: int = 128,
) -> None:
    classes = load_classes(taxonomy_path)
    with open(taxonomy_path, encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)
    taxonomy_cfg = taxonomy.get("config", {})

    with open(world_path, encoding="utf-8") as f:
        world = json.load(f)

    if eval_data_path:
        val_records = load_dataset(eval_data_path)
        print(f"Eval set: {len(val_records)} samples (from {eval_data_path})")
    else:
        records = load_dataset(data_path)
        _, val_records = stratified_split(records, val_ratio=0.2, seed=seed)
        print(f"Eval set: {len(val_records)} samples (20% val split, seed={seed})")
    print(f"Classes: {classes}")
    print(f"Model: {model_path}")

    session = build_session(model_path)

    # batch inference
    preds = []
    for i in range(0, len(val_records), batch_size):
        batch = val_records[i : i + batch_size]
        preds.extend(predict_batch(session, [r["text"] for r in batch]))

    # run axes
    results = {}
    results["axis1"], _ = axis1_per_class_f1(val_records, preds, classes, taxonomy_cfg)
    results["axis2"] = axis2_crosslingual(val_records, preds, classes, taxonomy_cfg)
    results["axis3"] = axis3_codeswitching(val_records, preds, classes)
    results["axis4"] = axis4_hard_negatives(val_records, preds, classes, world)
    results["axis5"] = axis5_length_buckets(val_records, preds, classes)
    lat_pass, med, p95 = axis6_latency(session, val_records[0]["text"])
    results["axis6"] = lat_pass

    # overall summary
    print("\n" + "=" * 50)
    print("EVAL SUMMARY")
    print("=" * 50)
    labels = {
        "axis1": "Per-class F1",
        "axis2": "Cross-lingual",
        "axis3": "Code-switching",
        "axis4": "Hard negatives",
        "axis5": "Length buckets",
        "axis6": f"Latency (median={med:.1f}ms p95={p95:.1f}ms)",
    }
    for k, label in labels.items():
        status = "PASS" if results[k] else "FAIL"
        print(f"  {label:<40} {status}")

    overall = "PASS" if all(results.values()) else "FAIL"
    print(f"\nOverall: {overall}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/charcnn_trained.onnx")
    parser.add_argument("--data", default="data/dataset.jsonl")
    parser.add_argument("--taxonomy", default="taxonomy.yaml")
    parser.add_argument("--world", default="data/world.json")
    parser.add_argument("--eval-data", default=None,
                        help="If set, use this file as the eval set instead of the val split")
    args = parser.parse_args()
    evaluate(
        model_path=args.model,
        data_path=args.data,
        taxonomy_path=args.taxonomy,
        world_path=args.world,
        eval_data_path=args.eval_data,
    )


if __name__ == "__main__":
    main()
