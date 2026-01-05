#!/bin/bash
set -e

# 1. Clean up old files
rm -rf site-packages tokenizer_config model.tflite action.zip
mkdir site-packages

echo "--- Installing lightweight dependencies ---"
# Install only what we need for inference (no full PyTorch/TensorFlow)
pip install tflite-runtime transformers numpy -t site-packages/

echo "--- Downloading Tokenizer and Model ---"
# We use a small python snippet to save the tokenizer and convert the model
python3 <<EOF
import os
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

model_id = "distilbert-base-uncased-finetuned-sst-2-english"

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("./tokenizer_config")

print("Downloading and converting model to TFLite (this may take a minute)...")
model = TFAutoModelForSequenceClassification.from_pretrained(model_id, from_pt=True)

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optimize for size (Quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
EOF

echo "--- Packaging Action ---"
# Zip everything including the Python script, model, and local libraries
zip -r action.zip __main__.py model.tflite tokenizer_config site-packages

echo "--- DONE! ---"
echo "Total package size: \$(du -h action.zip | cut -f1)"
echo "Deploy with: wsk action create distilbert-action action.zip --kind python:3.9 --memory 512"