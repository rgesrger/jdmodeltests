#!/bin/bash
set -e

echo "--- 1. Cleaning up ---"
rm -rf site-packages tokenizer_config model.tflite action.zip
mkdir site-packages

echo "--- 2. Installing minimum libraries ---"
# Only installing what is needed for the conversion
pip install tflite-runtime transformers numpy tensorflow-cpu -t site-packages/

echo "--- 3. Converting TinyBERT to TFLite (Tiny & Fast) ---"
export PYTHONPATH=$PYTHONPATH:$(pwd)/site-packages
python3 <<EOF
import os, sys
sys.path.append(os.path.join(os.getcwd(), "site-packages"))
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# Switching to TinyBERT - significantly smaller footprint
model_id = "huawei-noah/TinyBERT_General_4L_312D"

print("Downloading tiny tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("./tokenizer_config")

print("Downloading and converting TinyBERT...")
model = TFAutoModelForSequenceClassification.from_pretrained(model_id, from_pt=True)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
EOF

echo "--- 4. Aggressive Scrubbing ---"
# Remove the massive TensorFlow and other junk to save 200MB+
cd site-packages
rm -rf tensorflow* tensorboard* keras* google* _pywrap_tensorflow*
rm -rf *.dist-info *.egg-info
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
# Strip the transformers library to only the base
find . -name "tests" -type d -exec rm -rf {} +
cd ..

echo "--- 5. Final Zip ---"
zip -r action.zip __main__.py model.tflite tokenizer_config site-packages

echo "--- SUCCESS ---"
echo "Action size: $(du -h action.zip | cut -f1) (Target was < 48MB)"