#!/bin/bash
# Qwen3.5-9B via mlx-lm with speculative decoding (Apple Silicon native)
# Main: mlx-community/Qwen3.5-9B-4bit (~5GB)
# Draft: z-lab/Qwen3.5-9B-DFlash (~1GB)
# OpenAI-compatible API on port 3333

MAIN_MODEL="mlx-community/Qwen3.5-9B-4bit"
DRAFT_MODEL="z-lab/Qwen3.5-9B-DFlash"
PORT=${PORT:-3333}

echo "Launching Qwen3.5-9B (mlx 4bit) + speculative decoding on port $PORT..."

/Users/vlwkaos/.hermes/hermes-agent/venv/bin/mlx_lm.server \
  --model "$MAIN_MODEL" \
  --draft-model "$DRAFT_MODEL" \
  --num-draft-tokens 8 \
  --port "$PORT" \
  --host 127.0.0.1
