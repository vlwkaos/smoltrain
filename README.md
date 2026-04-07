# smoltrain

Train a tiny task-specific classifier model on the fly.

Given a goal, label set, and a few sample answers — generate synthetic training
data with a supervisor model (Claude, local LLM, or any OpenAI-compatible API),
fine-tune a sub-1B base model with LoRA, export to GGUF, and optionally serve
as a hot daemon for sub-50ms inference.

Built for [smolroute](https://github.com/vlwkaos/smolroute) but general enough
for any single-token classification task.

---

## How it works

```
smoltrain new routing \
  --goal "Classify an LLM request into one of these routing buckets" \
  --classes code,edit,chat,research \
  --base Qwen/Qwen3-0.6B

smoltrain gen routing --n 500 --supervisor claude   # generate synthetic (input, label) pairs
smoltrain train routing --epochs 3                  # LoRA fine-tune via mlx-lm
smoltrain eval routing                              # accuracy + confusion matrix
smoltrain export routing --format gguf              # export to .gguf for llama.cpp
smoltrain serve routing --port 8765                 # hot daemon (unix socket or TCP)
```

### Inference pattern (single forward pass)

The trained model is used as a classifier — NOT a text generator.
Classification is done by reading the logits of the FIRST generated token
and taking argmax (or softmax) over the label token IDs.

```
Input: "<system>You are a routing classifier...\n<user>{request}"
Output: argmax over logits[class_token_ids]  → label in ~10-50ms
```

This is the same mechanism as the Qwen3-Reranker in [ir](https://github.com/vlwkaos/ir)
(softmax over Yes/No), extended to N labels.

---

## Architecture

```
smoltrain/
├── src/
│   ├── main.rs          — CLI entry (clap)
│   ├── config.rs        — Project config (smoltrain.toml per task)
│   ├── gen/             — Training data generation
│   │   ├── mod.rs
│   │   ├── supervisor.rs    — Supervisor API client (Anthropic / OpenAI-compat)
│   │   └── dataset.rs       — JSONL dataset read/write
│   ├── train/           — Fine-tuning orchestration
│   │   ├── mod.rs
│   │   └── mlx.rs           — Invoke mlx-lm via subprocess
│   ├── eval/            — Accuracy / confusion matrix
│   │   └── mod.rs
│   ├── export/          — GGUF export via llama.cpp convert
│   │   └── mod.rs
│   └── serve/           — Hot daemon: load model once, classify in loop
│       ├── mod.rs
│       └── classifier.rs    — llama.cpp inference (logit-only, no generation)
├── scripts/
│   ├── train.py         — mlx-lm LoRA wrapper (called by smoltrain train)
│   └── export.py        — MLX → GGUF conversion
├── smoltrain.toml       — Example project config
└── Cargo.toml
```

---

## Supervisor data format

Training data is stored as JSONL at `~/.local/share/smoltrain/<task>/train.jsonl`:

```jsonl
{"input": "implement a binary search tree in rust", "label": "code"}
{"input": "fix the typo in README.md line 12", "label": "edit"}
{"input": "what is the difference between vec and slice", "label": "chat"}
{"input": "find papers on retrieval augmented generation", "label": "research"}
```

The supervisor model is prompted to generate diverse, realistic examples per class.

---

## Dependencies

- Rust 1.80+
- Python 3.10+ with `mlx-lm` (`pip install mlx-lm`)
- llama.cpp (`brew install llama.cpp`) for GGUF export
- An Anthropic API key OR a local OpenAI-compatible server for supervision

---

## Relation to ir

smoltrain reuses the llama.cpp classifier pattern from
[ir/src/llm/reranker.rs](https://github.com/vlwkaos/ir) — same logit-softmax
approach, generalized to N classes instead of Yes/No.

The `serve` daemon keeps the model loaded in memory (like ir's hot model path),
eliminating the ~400ms cold-load penalty and getting inference to 10-50ms.
