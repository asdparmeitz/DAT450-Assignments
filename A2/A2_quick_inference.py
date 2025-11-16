"""
Quick inference script - simple example of using the trained model.
"""
import torch
from A2_inference import load_model_and_tokenizer, predict_next, generate_text

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 
                     'mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Loading model on {device}...")
model, tokenizer = load_model_and_tokenizer()
model.to(device)
model.eval()

print("\n" + "="*60)
print("QUICK INFERENCE")
print("="*60)

# Example: Predict next tokens
prompt = "She lives in San"
print(f"\nPrompt: '{prompt}'")
top5 = predict_next(tokenizer, model, prompt, device, k=5)
print(f"Top 5 predictions: {top5}")

# Example: Generate text
print(f"\nGenerating text from: '{prompt}'")
generated = generate_text(tokenizer, model, prompt, device, max_length=20, temperature=0.8)
print(f"Generated: {generated}")

print("\nDone!")

