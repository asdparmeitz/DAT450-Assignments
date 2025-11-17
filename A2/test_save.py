# test_save.py
import torch
from A2_skeleton import A2Transformer, A2ModelConfig
import json
import os

# Load your existing config
config_path = 'A2/trainer_output/config.json'
with open(config_path, 'r') as f:
    config_dict = json.load(f)

config = A2ModelConfig(**config_dict)

# Create a dummy model (random weights, but tests if save works)
print("Creating test model...")
model = A2Transformer(config)
model.eval()

# Test saving
output_dir = 'A2/trainer_output_test'
os.makedirs(output_dir, exist_ok=True)

print(f"Testing save to {output_dir}...")
try:
    model.save_pretrained(output_dir)
    print("✓ Save successful!")
    
    # Check what was saved
    saved_files = os.listdir(output_dir)
    print(f"Files saved: {saved_files}")
    
    has_weights = any(f.endswith(('.safetensors', '.bin')) for f in saved_files)
    if has_weights:
        print("✓ Model weights file found!")
    else:
        print("✗ No weights file found!")
        
except Exception as e:
    print(f"✗ Save failed: {e}")
    import traceback
    traceback.print_exc()