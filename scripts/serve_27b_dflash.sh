#!/bin/bash
# Qwen3.5-27B (4bit) + DFlash speculative decoding
# ~14GB main (4bit) + 1.7GB draft = ~16GB total — fits in 24GB unified
# High quality, still 6x faster than baseline 27B
# Serves OpenAI-compatible API on port 8001

set -e

MAIN_MODEL="Qwen/Qwen3.5-27B"
DRAFT_MODEL="z-lab/Qwen3.5-27B-DFlash"
PORT=${PORT:-3334}

echo "Launching Qwen3.5-27B (4bit) + DFlash on port $PORT..."
echo "Main:  $MAIN_MODEL (4bit quantized)"
echo "Draft: $DRAFT_MODEL"
echo ""

# vLLM with MPS backend + 4bit quantization (Apple Silicon, 24GB)
vllm serve "$MAIN_MODEL" \
  --speculative-config "{\"method\": \"dflash\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 15}" \
  --quantization bitsandbytes \
  --load-in-4bit \
  --device mps \
  --dtype bfloat16 \
  --trust-remote-code \
  --port "$PORT" \
  --gpu-memory-utilization 0.85

# Alternative: AWQ pre-quantized (better quality than bitsandbytes 4bit)
# First find/create AWQ: search HF for "Qwen3.5-27B-AWQ"
# Then:
# vllm serve "MODEL_AWQ_ID" \
#   --speculative-config "{\"method\": \"dflash\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 15}" \
#   --quantization awq \
#   --device mps \
#   --trust-remote-code \
#   --port "$PORT"

# SGLang alternative (CUDA only):
# python -m sglang.launch_server \
#   --model-path "$MAIN_MODEL" \
#   --speculative-algorithm DFLASH \
#   --speculative-draft-model-path "$DRAFT_MODEL" \
#   --tp-size 1 --dtype bfloat16 \
#   --quantization awq \
#   --attention-backend fa3 \
#   --trust-remote-code \
#   --port "$PORT"
