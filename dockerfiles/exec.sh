#!/bin/sh
set -e

# echo "[DEBUG] /action/exec start"
# echo "[DEBUG] CWD: $(pwd)"
# echo "[DEBUG] /app contents:"
ls -lh /app

# Skip stdin for testing
# INPUT=$(cat)
# echo "[DEBUG] Received input JSON: $INPUT"

# echo "[DEBUG] Running model..."
OUTPUT=$(/app/distilgpt2_infer /app/distilgpt2.onnx)
echo "[DEBUG] Model finished"

# Wrap result in JSON
echo "{\"result\": \"$OUTPUT\"}"
