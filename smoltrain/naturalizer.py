"""Naturalize a JSONL dataset via LLM rephrasing.

Backends:
  --local   Use local mlx_lm.server (default port 8766). No API key needed.
            Start server first:
              mlx_lm.server --model mlx-community/Qwen3.5-4B-MLX-4bit \\
                --port 8766 --chat-template-args '{"enable_thinking": false}' \\
                --decode-concurrency 8 --max-tokens 120
  default   OpenRouter (OPENROUTER_API_KEY or keychain: openrouter)

Usage:
    python -m smoltrain.naturalizer \\
        --input data/dataset.jsonl \\
        --output data/naturalized.jsonl \\
        --n 300 --seed 99 --local --workers 8
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock


LOCAL_URL   = "http://localhost:8766/v1"
LOCAL_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
REMOTE_MODEL = "meta-llama/llama-3.1-8b-instruct"

SYSTEM_PROMPT = (
    "Rewrite the given text as something a real developer would type in a chat interface. "
    "Rules: "
    "1. Output ONLY the rewritten message — no quotes, no explanation, no context description. "
    "2. Keep the same language (Korean stays Korean, English stays English, mixed stays mixed). "
    "3. Preserve the original intent exactly — do not change what is being asked. "
    "4. Sound like a real person typing quickly — casual, direct, sometimes incomplete sentences are fine. "
    "5. Never start with 'Sure', 'Of course', or any acknowledgment."
)


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_done(output_path: str) -> set[str]:
    """Return set of original texts already written (for resume)."""
    done = set()
    p = Path(output_path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    done.add(rec.get("original_text", ""))
                except Exception:
                    pass
    return done


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_label: dict[str, list] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)

    num_classes = len(by_label)
    per_class = n // num_classes
    remainder = n - per_class * num_classes

    sampled = []
    for i, (label, items) in enumerate(sorted(by_label.items())):
        count = per_class + (1 if i < remainder else 0)
        pool = items[:]
        rng.shuffle(pool)
        sampled.extend(pool[:count])

    rng.shuffle(sampled)
    return sampled


def rephrase(client, text: str, model: str) -> str:
    """Single LLM call. Caller handles retry."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.9,
        max_tokens=120,
    )
    content = resp.choices[0].message.content
    if not content:
        raise ValueError("empty response")
    return content.strip()


def process_record(client, model, record, idx, total, retries=4):
    """Rephrase one record with retry. Returns (idx, output_dict)."""
    import openai as _openai
    original = record["text"]
    for attempt in range(retries):
        try:
            naturalized = rephrase(client, original, model)
            return idx, {**record, "text": naturalized, "original_text": original}
        except _openai.RateLimitError:
            wait = 5 * (attempt + 1)
            print(f"  [{idx}/{total}] 429, retry in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  [{idx}/{total}] err: {e}, keeping original")
            return idx, {**record, "text": original, "original_text": original}
    print(f"  [{idx}/{total}] all retries failed, keeping original")
    return idx, {**record, "text": original, "original_text": original}


def naturalize(
    input_path: str,
    output_path: str,
    n: int,
    seed: int,
    local: bool = False,
    workers: int = 8,
) -> None:
    from openai import OpenAI

    if local:
        client = OpenAI(api_key="local", base_url=LOCAL_URL)
        model = LOCAL_MODEL
        print(f"Backend: local mlx_lm.server @ {LOCAL_URL}  workers={workers}")
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            import subprocess
            r = subprocess.run(
                ["security", "find-generic-password", "-a", os.environ["USER"],
                 "-s", "openrouter", "-w"],
                capture_output=True, text=True,
            )
            api_key = r.stdout.strip()
        if not api_key:
            print("Error: no OpenRouter key. Run: kc add smoltrain.openrouter")
            raise SystemExit(1)
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        client = OpenAI(api_key=api_key, base_url=base_url)
        model = REMOTE_MODEL
        workers = min(workers, 4)  # be gentle with remote API
        print(f"Backend: OpenRouter  model={model}  workers={workers}")

    records = load_records(input_path)
    samples = stratified_sample(records, n, seed)
    total = len(samples)

    # Resume: skip already-done records
    done = load_done(output_path)
    pending = [(i + 1, r) for i, r in enumerate(samples) if r["text"] not in done]
    skipped = total - len(pending)
    if skipped:
        print(f"Resuming: {skipped} already done, {len(pending)} remaining")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_lock = Lock()
    completed = skipped
    t0 = time.time()

    with open(out_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(process_record, client, model, record, idx, total): idx
                for idx, record in pending
            }
            for future in as_completed(futures):
                idx, out = future.result()
                with write_lock:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                    completed += 1
                    elapsed = time.time() - t0
                    rate = (completed - skipped) / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(f"[{completed}/{total}] ok  {rate:.1f} rec/s  ETA {eta:.0f}s")

    print(f"Done. Wrote {total} samples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/dataset.jsonl")
    parser.add_argument("--output", default="data/naturalized.jsonl")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--local", action="store_true",
                        help="Use local mlx_lm.server on port 8766 (no API key)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent workers (default 8 for local, 4 for remote)")
    args = parser.parse_args()
    naturalize(args.input, args.output, args.n, args.seed,
               local=args.local, workers=args.workers)


if __name__ == "__main__":
    main()
