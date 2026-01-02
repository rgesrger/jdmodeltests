#!/bin/sh
set -e

# Read JSON input from OpenWhisk
INPUT=$(cat)

# Currently ignoring INPUT; just run model
OUTPUT=$(/app/distilgpt2_infer /app/distilgpt2.onnx)

# Wrap result in JSON for OpenWhisk
echo "{\"result\": \"$OUTPUT\"}"
