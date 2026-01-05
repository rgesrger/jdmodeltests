#!/bin/bash
set -e

echo "--- 1. Cleaning up ---"
rm -rf site-packages tokenizer_config model.tflite action.zip
mkdir site-packages

echo "--- 2. Installing runtime only (No Torch/No TF) ---"
# We only install the lightweight runtime and tokenizer
pip install tflite-runtime tokenizers numpy -t site-packages/

echo "--- 3. Downloading Pre-converted Tiny Model ---"
# We download the model and vocab directly using curl to avoid the 'torch' error
curl -L https://huggingface.co/kimil79/tinybert-6l-768d-squad2-tflite/resolve/main/model.tflite -o model.tflite

mkdir -p tokenizer_config
curl -L https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D/resolve/main/vocab.txt -o tokenizer_config/vocab.txt

echo "--- 4. Creating Action Code ---"
# Ensure __main__.py is ready (see below)

echo "--- 5. Final Zip ---"
# Aggressive cleanup of site-packages to save space
find site-packages -name "__pycache__" -type d -exec rm -rf {} +
zip -r action.zip __main__.py model.tflite tokenizer_config site-packages

echo "--- SUCCESS ---"
echo "Final Zip Size: $(du -h action.zip | cut -f1)"