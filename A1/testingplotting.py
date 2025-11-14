
from A1_skeleton import A1RNNModel, A1Tokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import torch, nltk, pickle
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
import sys, time, os
from torch.utils.data import Subset


model = A1RNNModel.from_pretrained("trainer_output")
tokenizer = A1Tokenizer.from_file("tokenizer.pkl")
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
test_path = os.path.join(SCRIPT_DIR, "test.txt")


test_dataset = load_dataset('text', data_files={'test': test_path})
test_dataset = test_dataset.filter(lambda x: x['text'].strip() != '')

# Evaluate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
loss_func = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
V = model.config.vocab_size

test_loss_sum, test_tok = 0.0, 0
with torch.no_grad():
    for batch in test_loader:
        texts = batch["text"]
        enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        X, Y = ids[:, :-1], ids[:, 1:]
        logits = model(X)
        loss = loss_func(logits.reshape(-1, V), Y.reshape(-1))
        num_tokens = (Y != tokenizer.pad_token_id).sum().item()
        test_loss_sum += loss.item() * num_tokens  ## OBS DUBBELCHECK HERE!!!
        test_tok += num_tokens

test_ce = test_loss_sum / max(1, test_tok)
test_ppl = float(np.exp(test_ce))
print(f"Test CE={test_ce:.4f} PPL={test_ppl:.1f}")

log_file = "logs/testingplotting-84945.out"

with open(log_file, 'r') as f:
    lines = f.readlines()

epochs, train_ce, train_ppl, val_ce, val_ppl = [], [], [], [], []

for line in lines:
    match = re.search(r'Epoch (\d+): train CE=([\d.]+) PPL=([\d.]+) \| val CE=([\d.]+) PPL=([\d.]+)', line)
    if match:
        epochs.append(int(match.group(1)))
        train_ce.append(float(match.group(2)))
        train_ppl.append(float(match.group(3)))
        val_ce.append(float(match.group(4)))
        val_ppl.append(float(match.group(5)))

# Plot
fig, (ax1, ax2) = plt.subplot(1, 2, figsize=(12, 5))
ax1.plot(epochs, train_ce, 'b-o', label='Train')
ax1.plot(epochs, val_ce, 'r-o', label='Validation')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.set_title('Training vs Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, train_ppl, 'b-o', label='Train')
ax2.plot(epochs, val_ppl, 'r-o', label='Validation')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Perplexity')
ax2.set_title('Training vs Validation Perplexity')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
print("Saved training_curves.png")


def predict_next(tokenizer: A1Tokenizer, model: PreTrainedModel, text: str, device: torch.device, k: int = 5):
    model.eval()
    with torch.no_grad():
        enc = tokenizer(text, return_tensors='pt', padding=False, truncation=True)
        ids = enc["input_ids"].to(device)
        X = ids[:, :-1]
        logits = model(X)
        last_logits = logits[:, -1, :]
        topk = torch.topk(last_logits, k=k, dim=-1)
        idxs = topk.indices[0].tolist()
        return [tokenizer.id2word.get(i, tokenizer.unk_token) for i in idxs]


# Test predictions
test_prompts = [
    "She lives in San",
    "The capital of France is",
    "In the beginning",
    "The quick brown"
]

for prompt in test_prompts:
    top5 = predict_next(tokenizer, model, prompt, device, k=5)
    print(f"{prompt} → {top5}")