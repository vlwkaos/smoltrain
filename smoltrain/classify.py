"""ONNX-based classifier for fast inference."""
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from . import config as cfg_mod


class OnnxClassifier:
    def __init__(self, cfg):
        tok_path = cfg_mod.tokenizer_path(cfg.name)
        onnx_p = cfg_mod.onnx_path(cfg.name)

        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tok_path}. Run: smoltrain export {cfg.name}")
        if not onnx_p.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_p}. Run: smoltrain export {cfg.name}")

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.session = ort.InferenceSession(str(onnx_p))
        self.classes = cfg.classes

    def classify(self, text: str) -> str:
        enc = self.tokenizer.encode(text)
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        logits = self.session.run(
            ["logits"], {"input_ids": ids, "attention_mask": mask}
        )[0]
        return self.classes[int(np.argmax(logits[0]))]
