#!/bin/sh
set -e

echo "[DEBUG] /action/exec started"
echo "[DEBUG] CWD: $(pwd)"
echo "[DEBUG] /app contents:"
ls -l /app

# Read JSON input from OpenWhisk
INPUT=$(cat)
echo "[DEBUG] Received input JSON: $INPUT"

# Run the model
OUTPUT=$(/app/distilgpt2_infer /app/distilgpt2.onnx)
echo "[DEBUG] Model output: $OUTPUT"

# Wrap output in JSON for OpenWhisk
echo "{\"result\": \"$OUTPUT\"}"
echo "[DEBUG] /action/exec finished"
