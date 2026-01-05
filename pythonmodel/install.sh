#!/bin/bash
set -e

echo "--- 1. Cleaning up old files ---"
rm -rf site-packages tokenizer_config model.tflite action.zip
mkdir site-packages

echo "--- 2. Installing libraries into site-packages ---"
# -t site-packages tells pip to put the code here instead of in your system folders
pip install tflite-runtime transformers numpy -t site-packages/

echo "--- 3. Downloading and Converting DistilBERT to TFLite ---"
# This python block creates the model and tokenizer files needed for the zip
python3 <<EOF
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import os

model_id = "distilbert-base-uncased-finetuned-sst-2-english"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("./tokenizer_config")

print("Converting model (this stays under 100MB)...")
model = TFAutoModelForSequenceClassification.from_pretrained(model_id, from_pt=True)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
EOF

echo "--- 4. Creating action.zip ---"
# This bundles everything OpenWhisk needs into one file
zip -r action.zip __main__.py model.tflite tokenizer_config site-packages

echo "--- SUCCESS ---"
echo "Your 'action.zip' is ready. Total size: $(du -h action.zip | cut -f1)"