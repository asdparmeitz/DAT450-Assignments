# At the end of A2_train.py, after training completes:

# Test predictions (similar to A1)
from A2_inference import predict_next

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_prompts = [
    "She lives in San",
    "The capital of France is",
    "In the beginning"
]

print("\n" + "="*60)
print("TESTING PREDICTIONS")
print("="*60)

for prompt in test_prompts:
    top5 = predict_next(tokenizer, model, prompt, device, k=5)
    print(f"'{prompt}' → {top5}")