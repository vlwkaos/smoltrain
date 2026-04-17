#!/bin/bash
# Benchmark tok/sec against local mlx server
# Usage: bash bench_tokspeed.sh [port] [num_requests]

PORT=${1:-3333}
N=${2:-20}
MODEL="mlx-community/Qwen3.5-9B-4bit"

echo "Benchmarking port $PORT, $N requests..."
echo ""

/Users/vlwkaos/.hermes/hermes-agent/venv/bin/python3 - <<'EOF'
import sys, time, json, statistics
from openai import OpenAI

port = int(sys.argv[1]) if len(sys.argv) > 1 else 3333
n = int(sys.argv[2]) if len(sys.argv) > 2 else 20

client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local")

prompts = [
    "rename the variable x to something descriptive",
    "왜 이 코드가 느린지 분석해줘",
    "can you explain what this regex does",
    "fix the typo in this comment",
    "이 함수 리팩토링 해줘 — 중복 로직이 너무 많아",
    "what does n+1 query mean",
    "deploy this to staging and run tests",
    "this import is unused, remove it",
    "코드 리뷰 해줘 전체적으로",
    "what is the difference between map and flatMap",
]

tok_rates = []
total_toks = 0
t_total = time.time()

for i in range(n):
    prompt = prompts[i % len(prompts)]
    t0 = time.time()
    resp = client.chat.completions.create(
        model="mlx-community/Qwen3.5-9B-4bit",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=80,
        temperature=0.7,
    )
    elapsed = time.time() - t0
    out_toks = resp.usage.completion_tokens if resp.usage else 0
    total_toks += out_toks
    rate = out_toks / elapsed if elapsed > 0 else 0
    tok_rates.append(rate)
    print(f"  [{i+1}/{n}] {out_toks} tok in {elapsed:.2f}s = {rate:.1f} tok/s")

wall = time.time() - t_total
print()
print("=" * 40)
print(f"  Requests:    {n}")
print(f"  Total toks:  {total_toks}")
print(f"  Wall time:   {wall:.1f}s")
print(f"  Median:      {statistics.median(tok_rates):.1f} tok/s")
print(f"  Mean:        {statistics.mean(tok_rates):.1f} tok/s")
print(f"  Min/Max:     {min(tok_rates):.1f} / {max(tok_rates):.1f} tok/s")
print("=" * 40)
EOF
