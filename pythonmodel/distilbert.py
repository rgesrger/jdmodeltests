import os
import json
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.pipelines import pipeline

# Global variables to cache the model across invocations (Warm Start)
classifier = None

def main(args):
    global classifier
    
    # 1. Initialize the model if it's the first time (Cold Start)
    if classifier is None:
        model_path = "./model"  # Path inside your action ZIP
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = ORTModelForSequenceClassification.from_pretrained(model_path)
        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # 2. Get input from the OpenWhisk activation
    text = args.get("text", "Hello! This is a default test string.")
    
    # 3. Perform Inference
    result = classifier(text)

    # 4. Return results as a dictionary
    return {
        "status": "success",
        "input": text,
        "prediction": result[0]
    }