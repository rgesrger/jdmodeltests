#!/bin/bash
set -e

echo "--- 1. Cleaning up old files ---"
rm -rf site-packages tokenizer_config model.tflite action.zip
mkdir site-packages

echo "--- 2. Installing libraries into site-packages ---"
# We add tensorflow-cpu here so the conversion script can run
pip install tflite-runtime transformers numpy tensorflow-cpu -t site-packages/

echo "--- 3. Downloading and Converting DistilBERT to TFLite ---"
export PYTHONPATH=$PYTHONPATH:$(pwd)/site-packages

python3 <<EOF
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "site-packages"))

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

# Using the standard base model which has native TF weights available
model_id = "distilbert-base-uncased-finetuned-sst-2-english"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("./tokenizer_config")

print("Downloading and converting model (No PyTorch needed)...")
# Note: we removed from_pt=True to avoid the Torch dependency
model = TFAutoModelForSequenceClassification.from_pretrained(model_id)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Required for some Transformer operations in TFLite
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS
]
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
EOF

echo "--- 4. Creating action.zip ---"
# We exclude the massive tensorflow library from the zip to stay under 500MB
# tflite-runtime is all we need for the actual action
zip -r action.zip __main__.py model.tflite tokenizer_config site-packages -x "site-packages/tensorflow/*"

echo "--- SUCCESS ---"
echo "Your 'action.zip' is ready. Size: $(du -h action.zip | cut -f1)"