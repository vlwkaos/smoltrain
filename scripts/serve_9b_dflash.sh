#!/bin/bash
# Qwen3.5-9B + DFlash speculative decoding
# ~10.7GB bf16 total (9.7GB main + 1.0GB draft) — sweet spot for 24GB unified
# Serves OpenAI-compatible API on port 3333

set -e

MAIN_MODEL="Qwen/Qwen3.5-9B"
DRAFT_MODEL="z-lab/Qwen3.5-9B-DFlash"
PORT=${PORT:-3333}

echo "Launching Qwen3.5-9B + DFlash on port $PORT..."
echo "Main:  $MAIN_MODEL"
echo "Draft: $DRAFT_MODEL"
echo ""

# vLLM with MPS backend (Apple Silicon)
vllm serve "$MAIN_MODEL" \
  --speculative-config "{\"method\": \"dflash\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 15}" \
  --device mps \
  --dtype bfloat16 \
  --trust-remote-code \
  --port "$PORT" \
  --gpu-memory-utilization 0.85

# Fallback: standard speculative decoding if DFlash fails on MPS
# vllm serve "$MAIN_MODEL" \
#   --speculative-model "$DRAFT_MODEL" \
#   --num-speculative-tokens 5 \
#   --device mps \
#   --dtype bfloat16 \
#   --trust-remote-code \
#   --port "$PORT"

# SGLang (CUDA only):
# python -m sglang.launch_server \
#   --model-path "$MAIN_MODEL" \
#   --speculative-algorithm DFLASH \
#   --speculative-draft-model-path "$DRAFT_MODEL" \
#   --tp-size 1 --dtype bfloat16 \
#   --attention-backend fa3 \
#   --trust-remote-code \
#   --port "$PORT"
