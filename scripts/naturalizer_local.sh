#!/bin/bash
# Run naturalizer against local DFlash server instead of OpenRouter
# Start serve_4b_dflash.sh or serve_27b_dflash.sh first

PORT=${1:-3333}
INPUT=${2:-data/eval_raw.jsonl}
OUTPUT=${3:-data/naturalized_eval.jsonl}
N=${4:-100}

echo "Using local server on port $PORT"
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo "N:      $N"
echo ""

LOCAL_API_KEY="local" \
OPENROUTER_BASE_URL="http://localhost:$PORT/v1" \
python -m smoltrain.naturalizer \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --n "$N" \
  --seed 99
