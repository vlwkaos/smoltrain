#!/usr/bin/env python3
"""Export fine-tuned model to ONNX INT8."""

import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seq-len", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    seq_len = args.seq_len
    dummy_input_ids = torch.zeros(1, seq_len, dtype=torch.long)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(1, seq_len, dtype=torch.long)

    onnx_fp32 = output_dir / "model_fp32.onnx"
    onnx_int8 = output_dir / "model_int8.onnx"

    print("exporting to ONNX FP32...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_mask, dummy_token_type_ids),
        str(onnx_fp32),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"FP32 size: {onnx_fp32.stat().st_size / 1024 / 1024:.1f} MB")

    print("quantizing to INT8...")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        str(onnx_fp32),
        str(onnx_int8),
        weight_type=QuantType.QInt8,
    )
    print(f"INT8 size: {onnx_int8.stat().st_size / 1024 / 1024:.1f} MB")

    # copy tokenizer.json
    tokenizer_src = model_dir / "tokenizer.json"
    if tokenizer_src.exists():
        shutil.copy(tokenizer_src, output_dir / "tokenizer.json")
        print("copied tokenizer.json")
    else:
        tokenizer.save_pretrained(str(output_dir))
        print("saved tokenizer from HuggingFace")

    # cleanup FP32
    onnx_fp32.unlink()
    print(f"done. INT8 model at {onnx_int8}")


if __name__ == "__main__":
    main()
