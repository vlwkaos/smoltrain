#!/bin/bash
# Qwen3.5-4B + DFlash speculative decoding
# ~5.2GB total (4.7GB main + 0.5GB draft)
# Fast, lower quality — good for bulk generation tasks (naturalizer etc.)
# Serves OpenAI-compatible API on port 8000

set -e

MAIN_MODEL="Qwen/Qwen3.5-4B"
DRAFT_MODEL="z-lab/Qwen3.5-4B-DFlash"
PORT=${PORT:-3333}

echo "Launching Qwen3.5-4B + DFlash on port $PORT..."
echo "Main:  $MAIN_MODEL"
echo "Draft: $DRAFT_MODEL"
echo ""

# vLLM with MPS backend (Apple Silicon)
# Note: DFlash support on MPS is experimental — fallback to standard spec decoding if it fails
vllm serve "$MAIN_MODEL" \
  --speculative-config "{\"method\": \"dflash\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 15}" \
  --device mps \
  --dtype bfloat16 \
  --trust-remote-code \
  --port "$PORT" \
  --gpu-memory-utilization 0.80

# If DFlash fails on MPS, try standard speculative decoding:
# vllm serve "$MAIN_MODEL" \
#   --speculative-model "$DRAFT_MODEL" \
#   --num-speculative-tokens 5 \
#   --device mps \
#   --dtype bfloat16 \
#   --trust-remote-code \
#   --port "$PORT"

# SGLang alternative (requires CUDA — not for Apple Silicon):
# python -m sglang.launch_server \
#   --model-path "$MAIN_MODEL" \
#   --speculative-algorithm DFLASH \
#   --speculative-draft-model-path "$DRAFT_MODEL" \
#   --tp-size 1 --dtype bfloat16 \
#   --attention-backend fa3 \
#   --trust-remote-code \
#   --port "$PORT"
