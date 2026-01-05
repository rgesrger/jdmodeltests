#!/bin/bash
set -e

echo "--- 1. Cleaning up old files ---"
rm -rf site-packages tokenizer_config model.tflite action.zip
mkdir site-packages

echo "--- 2. Installing libraries into site-packages ---"
# Installing locally so we can bundle them in the zip
pip install tflite-runtime transformers numpy tensorflow-cpu -t site-packages/

echo "--- 3. Downloading and Converting DistilBERT to TFLite ---"
# This line tells the next command to look inside the folder we just made
export PYTHONPATH=$PYTHONPATH:$(pwd)/site-packages

python3 <<EOF
import os
import sys
# Double-check path inside python
sys.path.append(os.path.join(os.getcwd(), "site-packages"))

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

model_id = "distilbert-base-uncased-finetuned-sst-2-english"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("./tokenizer_config")

print("Converting model (this may take 1-2 minutes)...")
model = TFAutoModelForSequenceClassification.from_pretrained(model_id, from_pt=True)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
EOF

echo "--- 4. Creating action.zip ---"
zip -r action.zip __main__.py model.tflite tokenizer_config site-packages

echo "--- SUCCESS ---"
echo "Your 'action.zip' is ready. Total size: $(du -h action.zip | cut -f1)"