#!/bin/bash
set -e

echo "--- 1. Cleaning up ---"
rm -rf site-packages tokenizer_config model.tflite action.zip
mkdir site-packages

echo "--- 2. Installing libraries (Temporary full install for conversion) ---"
pip install tflite-runtime transformers numpy tensorflow-cpu -t site-packages/

echo "--- 3. Converting Model ---"
export PYTHONPATH=$PYTHONPATH:$(pwd)/site-packages
python3 <<EOF
import os, sys
sys.path.append(os.path.join(os.getcwd(), "site-packages"))
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

model_id = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained("./tokenizer_config")

model = TFAutoModelForSequenceClassification.from_pretrained(model_id)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Critical: Keep it simple to keep size small
tflite_model = converter.convert()
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
EOF

echo "--- 4. Scrubbing site-packages (The Secret Sauce) ---"
cd site-packages
# Remove the massive TensorFlow library (we only need tflite-runtime for inference)
rm -rf tensorflow* rm -rf tensorboard*
rm -rf keras*
rm -rf google*
# Remove all metadata, caches, and documentation
rm -rf *.dist-info *.egg-info
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
# Remove large unused files in transformers
rm -rf transformers/models/vit
rm -rf transformers/models/llama
# (Optional) remove more specific unused models if needed
cd ..

echo "--- 5. Final Zip ---"
# We only zip what is absolutely necessary
zip -r action.zip __main__.py model.tflite tokenizer_config site-packages

echo "--- SUCCESS ---"
echo "Target size: Under 48MB"
echo "Actual size: $(du -h action.zip | cut -f1)"