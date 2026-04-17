---
name: smoltrain-build
description: Interactively build a smoltrain CharCNN classifier end-to-end — ask the user what to classify, scaffold taxonomy.yaml, generate training data, train, eval, and optionally serve. Use when someone wants to train a classifier with smoltrain.
allowed-tools: Bash, Read, Write, Edit
---

# smoltrain-build

Guides the user through building a new CharCNN text classifier with smoltrain.
Result: trained ONNX model, <1ms inference, optionally served on a Unix socket.

## What smoltrain is

NOT a classifier. A classifier-BUILDING PIPELINE.
- CharCNN model (~14KB ONNX weights), byte-level encoding, no tokenizer needed
- Full pipeline: generate synthetic data → train → eval (6 axes) → ONNX export
- Serves via Unix socket daemon for ~0.47ms p50 inference
- Supports 2–N classes, multilingual (en/ko/mixed), hard negatives, naturalization

## Step 1 — Gather Requirements (ask these questions)

Ask the user all at once in a single message:

```
1. Task name (slug, e.g. sentiment, intent, lang-detect)
2. What are you classifying? (one sentence: "route LLM requests by complexity")
3. Classes — list each with a short description, e.g.:
     positive: clearly satisfied, happy outcome
     negative: frustrated, problem unresolved
     neutral: factual, no sentiment
4. Languages: en only / en+ko / en+ko+mixed?
5. Samples per class: 200 (fast/prototype) / 500 (good) / 1000 (production)?
6. Supervisor for data gen: claude-oauth (default) / groq / openai / local:<url>?
7. Work directory (default: ~/ws/smoltrain-tasks/<task_name>)?
8. Run serve daemon after training? (y/n)
```

Stop and wait for answers. Don't proceed until you have 1–3 at minimum.

## Step 2 — Scaffold

Write a `/tmp/smoltrain-classes.yaml` with the classes block (indented under `classes:`):

```yaml
  <class1>:
    description: "<description>"
  <class2>:
    description: "<description>"
```

Then run scaffold script:

```bash
bash ~/ws/smoltrain/.claude/skills/smoltrain-build/scripts/scaffold.sh \
  "<work_dir>" "<task_name>" "<goal>" "<n_per_class>" "<langs>" \
  /tmp/smoltrain-classes.yaml
```

Verify `<work_dir>/taxonomy.yaml` looks correct. Show it to the user and ask to confirm before continuing.

## Step 3 — Generate Training Data

```bash
cd ~/ws/smoltrain

# World builder — generates template corpus from taxonomy (uses ANTHROPIC_API_KEY or OPENROUTER_API_KEY)
python -m smoltrain.world_builder \
  --taxonomy <work_dir>/taxonomy.yaml \
  --output <work_dir>/data/world.json

# Expander — procedural slot-filling (free, no LLM)
python -m smoltrain.expander \
  --world <work_dir>/data/world.json \
  --taxonomy <work_dir>/taxonomy.yaml \
  --output <work_dir>/data/dataset.jsonl

# Gen — supervisor LLM synthetic examples (fills gaps, hard negatives)
python -m smoltrain.gen \
  --taxonomy <work_dir>/taxonomy.yaml \
  --output <work_dir>/data/dataset.jsonl \
  --supervisor <supervisor> \
  --supervisor-model <model>   # e.g. claude-haiku-4-5, llama3-8b-8192

# Naturalizer — LLM rephrasing pass (optional but improves realism)
python -m smoltrain.naturalizer \
  --input <work_dir>/data/dataset.jsonl \
  --output <work_dir>/data/naturalized.jsonl \
  --n <n_per_class * n_classes * 0.2 rounded> \
  --local   # if using local omlx/mlx server on port 8766
```

Note: world_builder needs ANTHROPIC_API_KEY or OPENROUTER_API_KEY env var.
Naturalizer --local uses mlx_lm.server on port 8766 (Qwen3.5-4B-4bit).
If no local server, omit --local (uses OpenRouter, needs OPENROUTER_API_KEY).

## Step 4 — Build eval set

```bash
cd ~/ws/smoltrain

# Merge all generated data, reserve 20% as eval
python - <<'EOF'
import json, random
from pathlib import Path

work_dir = Path("<work_dir>")
files = [work_dir/"data/dataset.jsonl", work_dir/"data/naturalized.jsonl"]
records = []
for f in files:
    if f.exists():
        records += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]

random.seed(42)
random.shuffle(records)
split = int(len(records) * 0.8)
train, eval_ = records[:split], records[split:]

(work_dir/"data/train_full.jsonl").write_text("\n".join(json.dumps(r) for r in train))
(work_dir/"data/eval_full.jsonl").write_text("\n".join(json.dumps(r) for r in eval_))
print(f"train={len(train)}  eval={len(eval_)}")
EOF
```

## Step 5 — Train + Eval + Export (pipeline)

```bash
cd ~/ws/smoltrain
python -m smoltrain.pipeline \
  --data <work_dir>/data/dataset.jsonl,<work_dir>/data/naturalized.jsonl \
  --eval <work_dir>/data/eval_full.jsonl \
  --taxonomy <work_dir>/taxonomy.yaml \
  --epochs 15 \
  --out-dir <work_dir>/models
```

Pipeline runs 6 steps automatically:
1. Setup — load taxonomy + data files
2. Merge — combine all JSONL sources, show class distribution
3. Train — CharCNN with early stopping (patience=3), saves best checkpoint
4. Eval — per-class F1, cross-lingual, code-switching, hard negatives, length buckets
5. ONNX Export — exports to `<work_dir>/models/charcnn_trained.onnx`
6. Latency Check — 200 runs, reports p50/p95 ms

Pass thresholds: all-class F1 >= 0.85, p50 latency <= 50ms.
Show user the PIPELINE SUMMARY table at the end.

## Step 6 — Serve (optional)

If user said yes to serving:

```bash
# smoltrain serve uses config.py task system — for pipeline-trained models use classify.py directly
cd ~/ws/smoltrain
python -c "
from smoltrain.model import CharCNN, encode_text
import onnxruntime as ort, numpy as np, socket, os, threading, yaml

ONNX_PATH = '<work_dir>/models/charcnn_trained.onnx'
TAXONOMY   = '<work_dir>/taxonomy.yaml'
SOCK_PATH  = '/tmp/smoltrain-<task_name>.sock'

with open(TAXONOMY) as f:
    taxonomy = yaml.safe_load(f)
classes = list(taxonomy['classes'].keys())
session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

def classify(text):
    arr = encode_text(text).reshape(1,-1).astype('int64')
    logits = session.run(['logits'], {'input_ids': arr})[0]
    return classes[int(logits.argmax())]

if os.path.exists(SOCK_PATH): os.unlink(SOCK_PATH)
srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
srv.bind(SOCK_PATH); srv.listen(32)
print(f'serving {classes} on {SOCK_PATH}')
while True:
    conn, _ = srv.accept()
    def handle(c):
        with c.makefile('rwb') as f:
            for line in f:
                t = line.decode().strip()
                if t: f.write((classify(t)+'\n').encode()); f.flush()
    threading.Thread(target=handle, args=(conn,), daemon=True).start()
"
```

Test with:
```bash
echo "your test input" | nc -U /tmp/smoltrain-<task_name>.sock
```

## Pitfalls

- taxonomy.yaml classes block must be indented 2 spaces under `classes:` key
- world_builder.py needs ANTHROPIC_API_KEY or OPENROUTER_API_KEY — check before running
- naturalizer default uses OpenRouter; pass `--local` only if mlx_lm.server is running on 8766
- pipeline.py `--eval` default is `data/eval_full.jsonl` — always pass explicit path for non-routing tasks
- The CLI `smoltrain new/run` commands wire to old DistilBERT flow — use `python -m smoltrain.pipeline` directly
- eval_full.jsonl must exist before running pipeline — run Step 4 first
- If F1 < 0.85: add more samples per class or improve class descriptions in taxonomy.yaml
- If latency > 50ms on CPU: unlikely with CharCNN, but check for ORT version issues

## Quick reference — supervisors

| supervisor key | needs | notes |
|---|---|---|
| claude-oauth | macOS keychain or ANTHROPIC_TOKEN | uses Claude Code token |
| anthropic | ANTHROPIC_API_KEY | direct API key |
| groq | GROQ_API_KEY | fast, free tier available |
| openai | OPENAI_API_KEY | |
| local:<url> | local server running | e.g. local:http://localhost:8082/v1 |
