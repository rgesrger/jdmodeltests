import os
import sys

# Crucial: Tell Python to look in our local folder for dependencies
sys.path.append(os.path.join(os.getcwd(), "site-packages"))

import numpy as np
import tflite_runtime.interpreter as tflite
from transformers import AutoTokenizer

# Cache for warm starts
interpreter = None
tokenizer = None

def main(args):
    global interpreter, tokenizer
    
    # Initialize once
    if interpreter is None:
        tokenizer = AutoTokenizer.from_pretrained("./tokenizer_config")
        interpreter = tflite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()

    text = args.get("text", "I love using serverless functions!")
    
    # Tokenize
    inputs = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors="np")
    input_data = inputs['input_ids'].astype(np.int32)

    # Run Inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    logits = interpreter.get_tensor(output_details[0]['index'])
    prediction = int(np.argmax(logits))
    
    return {
        "sentiment": "POSITIVE" if prediction == 1 else "NEGATIVE",
        "input": text
    }