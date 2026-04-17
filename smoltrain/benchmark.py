"""Benchmark CharCNN inference via ONNX Runtime."""
import os
import time
import statistics
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from smoltrain.model import CharCNN

MODELS_DIR = Path(__file__).parent.parent / "models"
ONNX_PATH = MODELS_DIR / "charcnn_dummy.onnx"

TEXTS = [
    "hello",                                                              # 5
    "rename this variable",                                               # 20
    "can you rename this variable to something clearer",                  # 49
    "while reviewing the auth module I noticed the variable name does not match what it stores rename it",  # 98
    (
        "I was looking at the login flow and noticed several issues the variable name authData is misleading "
        "it actually holds the decoded token payload not the raw auth data can you rename it to tokenPayload "
        "throughout the file"
    ),                                                                    # ~218
]

ITERATIONS = 1000


def export_onnx(model: CharCNN, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, model.max_len, dtype=torch.long)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    onnx.checker.check_model(str(path))


def bench(session: ort.InferenceSession, ids: np.ndarray, n: int) -> tuple[float, float]:
    """Returns (median_ms, p95_ms)."""
    feed = {"input_ids": ids}
    # warmup
    for _ in range(10):
        session.run(None, feed)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    median = statistics.median(times)
    p95 = times[int(0.95 * n)]
    return median, p95


def main() -> None:
    model = CharCNN()
    model.eval()

    print("Exporting to ONNX...")
    export_onnx(model, ONNX_PATH)
    size_kb = ONNX_PATH.stat().st_size / 1024
    print(f"Exported: {ONNX_PATH}  ({size_kb:.1f} KB)\n")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(ONNX_PATH), sess_options=opts)

    header = f"{'Text length':>12}  {'Median ms':>10}  {'p95 ms':>8}  {'Model KB':>9}"
    print(header)
    print("-" * len(header))

    for text in TEXTS:
        ids = model.encode(text).numpy().astype(np.int64)
        median, p95 = bench(session, ids, ITERATIONS)
        print(f"{len(text):>12}  {median:>10.3f}  {p95:>8.3f}  {size_kb:>9.1f}")

    print()


if __name__ == "__main__":
    main()
