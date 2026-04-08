"""Export fine-tuned model to ONNX INT8."""
import shutil
from pathlib import Path

from . import config as cfg_mod


def run(cfg):
    model_path = cfg_mod.model_dir(cfg.name)
    onnx_out = cfg_mod.onnx_dir(cfg.name)
    onnx_out.mkdir(parents=True, exist_ok=True)

    fp32_path = onnx_out / "model.onnx"
    int8_path = cfg_mod.onnx_path(cfg.name)
    tok_src = model_path / "tokenizer.json"
    tok_dst = cfg_mod.tokenizer_path(cfg.name)

    # Try optimum first
    try:
        _export_with_optimum(cfg, model_path, fp32_path, int8_path)
    except Exception as e:
        print(f"optimum export failed ({e}), falling back to torch.onnx...")
        _export_with_torch(cfg, model_path, fp32_path, int8_path)

    # Copy tokenizer.json
    if tok_src.exists():
        shutil.copy2(tok_src, tok_dst)
        print(f"Tokenizer copied to {tok_dst}")
    else:
        print(f"Warning: {tok_src} not found, tokenizer not copied")

    print(f"ONNX INT8 model at {int8_path}")


def _export_with_optimum(cfg, model_path, fp32_path, int8_path):
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer

    print("Exporting with optimum...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        str(model_path), export=True
    )
    ort_model.save_pretrained(str(fp32_path.parent))

    # Find the exported model file
    exported = fp32_path.parent / "model.onnx"
    if not exported.exists():
        raise FileNotFoundError(f"optimum did not produce {exported}")

    print("Quantizing to INT8...")
    quantizer = ORTQuantizer.from_pretrained(str(fp32_path.parent))
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(fp32_path.parent), quantization_config=qconfig)

    # Find quantized file
    candidates = list(fp32_path.parent.glob("model_quantized.onnx")) + \
                 list(fp32_path.parent.glob("model_int8.onnx"))
    if candidates:
        shutil.copy2(candidates[0], int8_path)
    else:
        # fallback: just use fp32
        shutil.copy2(exported, int8_path)
        print("Warning: quantized file not found, using FP32 model")


def _export_with_torch(cfg, model_path, fp32_path, int8_path):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.onnx import export as onnx_export
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("Loading model for torch.onnx export...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        attn_implementation="eager",
    )
    model.eval()

    # Dummy inputs
    dummy_text = "example input for export"
    enc = tokenizer(dummy_text, return_tensors="pt", max_length=cfg.max_seq_len, truncation=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    print(f"Exporting FP32 ONNX to {fp32_path}...")
    with torch.no_grad():
        onnx_export(
            model,
            (input_ids, attention_mask),
            f=str(fp32_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
            opset_version=14,
        )

    print(f"Quantizing to INT8 at {int8_path}...")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    print("Quantization complete")
