import os
import sys
# Local dependencies
sys.path.append(os.path.join(os.getcwd(), "site-packages"))

import numpy as np
import tflite_runtime.interpreter as tflite
from transformers import AutoTokenizer

interpreter = None
tokenizer = None

def main(args):
    global interpreter, tokenizer
    
    if interpreter is None:
        tokenizer = AutoTokenizer.from_pretrained("./tokenizer_config")
        interpreter = tflite.Interpreter(model_path="model.tflite")
        interpreter.allocate_tensors()

    text = args.get("text", "This is a tiny but powerful model.")
    
    # Pre-process
    inputs = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors="np")
    input_data = inputs['input_ids'].astype(np.int32)

    # Predict
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    logits = interpreter.get_tensor(output_details[0]['index'])
    # Since TinyBERT-General isn't fine-tuned for sentiment out of the box,
    # it returns hidden states or generic logits. 
    # For a plug-and-play sentiment result, use: "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    
    return {
        "status": "success",
        "output_shape": str(logits.shape),
        "message": "TinyBERT is running!"
    }