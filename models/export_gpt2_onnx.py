from pathlib import Path
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Configuration ---
model_id = "gpt2"
output_filename = "gpt2.onnx"

# --- Setup ---
cache_dir = Path(os.environ.get("HF_HOME", Path.cwd() / "hf-cache"))
output_dir = Path(os.environ.get("ONNX_OUT", Path.cwd()))
cache_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / output_filename

print(f"Starting export for {model_id} (Generation with Cache) to {output_path}")

# 1. Load Model
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
model.eval()
model.config.use_cache = True # Required for optimized generation

# 2. Define Inputs for Single-Step Trace
num_layers = model.config.n_layer
num_heads = model.config.n_head
head_dim = model.config.n_embd // num_heads
past_seq_len = 10 # Dummy length for trace shape

input_ids = torch.ones((1, 1), dtype=torch.long)
attention_mask = torch.ones((1, 1), dtype=torch.long)

# Create dummy past_key_values (cache)
# NOTE: Logic is identical to DistilGPT2, only the size of n_layer/n_head changes
past_shape = [1, num_heads, past_seq_len, head_dim]
past_key_values = tuple(
    [(torch.rand(past_shape), torch.rand(past_shape))
     for _ in range(num_layers)]
)

# Flatten arguments for torch.onnx.export
flattened_past = tuple(t for layer in past_key_values for t in layer)
model_args = (input_ids, attention_mask) + flattened_past

# 3. Define names and dynamic axes for the traced graph (including cache)
input_names = ["input_ids", "attention_mask"]
output_names = ["logits"]
dynamic_axes = {"input_ids": {0: "batch"}, "logits": {0: "batch", 1: "sequence"}}

for i in range(num_layers):
    input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
    output_names.extend([f"present.{i}.key", f"present.{i}.value"])
    for name in [f"past_key_values.{i}.key", f"past_key_values.{i}.value", 
                 f"present.{i}.key", f"present.{i}.value"]:
        dynamic_axes[name] = {0: "batch", 2: "past_sequence_length"}

# 4. Export to ONNX
torch.onnx.export(
    model,
    model_args,
    str(output_path),
    input_names=input_names,
    output_names=output_names,
    opset_version=13,
    dynamic_axes=dynamic_axes,
    export_params=True,
)

print(f"✅ Exported GPT2 to {output_path}")