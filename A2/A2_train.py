import torch
import sys
import os
from torch.utils.data import DataLoader
import numpy as np
import pickle
import nltk
from datasets import load_dataset

# Ensure required NLTK tokenizers are available
nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

# Import the Transformer model
from A2_skeleton import A2Transformer, A2ModelConfig

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Points to A2 directory
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Points to base directory (parent of A1 and A2)
A1_DIR = os.path.join(BASE_DIR, 'A1')
train_path = os.path.join(A1_DIR, "train.txt")
val_path = os.path.join(A1_DIR, "val.txt")

# Define lowercase_tokenizer function - needed for pickle deserialization
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

# Make it available in __main__ module for pickle
import __main__
if not hasattr(__main__, 'lowercase_tokenizer'):
    __main__.lowercase_tokenizer = lowercase_tokenizer

# Load tokenizer class from A1
# A1.py now has __main__ guard, so importing it won't execute training code
sys.path.insert(0, A1_DIR)
print("Importing A1Tokenizer...", flush=True)
from A1 import A1Tokenizer
print("A1Tokenizer imported successfully", flush=True)

# Copy trainer and training arguments classes from A1
class TrainingArguments:
    def __init__(self):
        self.optim = 'adamw_torch'
        self.eval_strategy = 'epoch'
        self.use_cpu = False
        self.no_cuda = False
        self.learning_rate = 2e-3
        self.num_train_epochs = 3
        self.per_device_train_batch_size = 32
        self.per_device_eval_batch_size = 32
        self.output_dir = os.path.join(SCRIPT_DIR, 'trainer_output')

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        """Return the device to use for training, depending on the training arguments and the available backends."""
        if self.args.use_cpu:
            return torch.device('cpu')
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        if torch.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args

        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        V = self.model.config.vocab_size
        
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        optimizer = torch.optim.AdamW(
                        self.model.parameters(),
                        lr = args.learning_rate,
                        weight_decay=0.01,      # L2 regularization
                    )

        train_loader = DataLoader(self.train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        val_loader = DataLoader(self.eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)

        for epoch in range(args.num_train_epochs):
            self.model.train()
            train_loss_sum, train_tok = 0.0, 0

            for batch in train_loader:
                texts = batch["text"]
                enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)

                X = input_ids[:, :-1]
                Y = input_ids[:, 1:]

                logits = self.model(X)
                loss = loss_func(logits.reshape(-1, V), Y.reshape(-1))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    num_tokens = (Y != self.tokenizer.pad_token_id).sum().item()
                    train_loss_sum += loss.item() * num_tokens
                    train_tok += num_tokens

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_loss_sum, val_tok = 0.0, 0
                for batch in val_loader:
                    texts = batch["text"]
                    enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                    ids = enc["input_ids"].to(device)
                    Xv, Yv = ids[:, :-1], ids[:, 1:]
                    logits = self.model(Xv)
                    loss_v = loss_func(logits.reshape(-1, V), Yv.reshape(-1))
                    num_tokens = (Yv != self.tokenizer.pad_token_id).sum().item()
                    val_loss_sum += loss_v.item() * num_tokens
                    val_tok += num_tokens

            train_ce = train_loss_sum / max(1, train_tok)
            val_ce = val_loss_sum / max(1, val_tok)
            train_ppl = float(np.exp(train_ce))
            val_ppl = float(np.exp(val_ce))

            print(f"Epoch {epoch+1}: train CE={train_ce:.4f} PPL={train_ppl:.1f} | val CE={val_ce:.4f} PPL={val_ppl:.1f}")
            self.model.train()

        print(f'Saving to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)


# Load dataset
dataset = load_dataset('text', data_files={'train': train_path, 'val': val_path})
dataset = dataset.filter(lambda x: x['text'].strip() != '')

train_dataset = dataset["train"]
eval_dataset = dataset["val"]

# Load tokenizer from A1
# Handle pickle loading with lowercase_tokenizer available
try:
    tokenizer = A1Tokenizer.from_file(os.path.join(A1_DIR, 'tokenizer.pkl'))
except (AttributeError, ModuleNotFoundError):
    # Fallback: direct pickle load
    with open(os.path.join(A1_DIR, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
print(f"Loaded tokenizer with vocab_size: {len(tokenizer)}", flush=True)

# Create Transformer configuration
# Using a small Transformer as recommended:
# - 2 layers (small model)
# - hidden_size: 256 (smaller than A1's 512 to keep it manageable)
# - 4 attention heads (256 / 4 = 64 head_dim)
# - intermediate_size: 4x hidden_size for SwiGLU (1024)
config = A2ModelConfig(
    vocab_size=len(tokenizer),
    hidden_size=256,
    intermediate_size=1024,  # 4x hidden_size for SwiGLU
    num_attention_heads=4,
    num_hidden_layers=2,
    rope_theta=10000.0,
    hidden_act='silu',
    max_position_embeddings=256,  # Same as tokenizer's model_max_length
    rms_norm_eps=1e-6
)

print(f"\nTransformer Configuration:", flush=True)
print(f"  vocab_size: {config.vocab_size}", flush=True)
print(f"  hidden_size: {config.hidden_size}", flush=True)
print(f"  intermediate_size: {config.intermediate_size}", flush=True)
print(f"  num_attention_heads: {config.num_attention_heads}", flush=True)
print(f"  num_hidden_layers: {config.num_hidden_layers}", flush=True)
print(f"  head_dim: {config.hidden_size // config.num_attention_heads}", flush=True)

# Create Transformer model
model = A2Transformer(config)
print(f"\nModel created with {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)

# Verify this is a Transformer, not RNN
assert isinstance(model, A2Transformer), "ERROR: Model is not A2Transformer!"
assert hasattr(model, 'layers'), "ERROR: Model missing Transformer layers!"
assert hasattr(model, 'rotary_emb'), "ERROR: Model missing RoPE embeddings!"
print("✓ Verified: This is an A2Transformer model", flush=True)

# Training arguments
args = TrainingArguments()

# Verify output directory
print(f"\nModel will be saved to: {args.output_dir}", flush=True)
assert 'A2' in args.output_dir or 'a2' in args.output_dir.lower(), \
    f"WARNING: Output directory {args.output_dir} doesn't seem to be in A2 folder!"

# Create trainer
trainer = A1Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Verify trainer has the Transformer model
assert isinstance(trainer.model, A2Transformer), "ERROR: Trainer has wrong model type!"
print("✓ Verified: Trainer contains A2Transformer", flush=True)

# Train the model
print("\n" + "="*60, flush=True)
print("STARTING TRANSFORMER TRAINING", flush=True)
print("="*60 + "\n", flush=True)
trainer.train()

print("\n" + "="*60, flush=True)
print("TRANSFORMER TRAINING COMPLETED!", flush=True)
print("="*60, flush=True)

# Verify the saved model
saved_config_path = os.path.join(args.output_dir, 'config.json')
if os.path.exists(saved_config_path):
    import json
    with open(saved_config_path, 'r') as f:
        saved_config = json.load(f)
    if saved_config.get('architectures', [None])[0] == 'A2Transformer':
        print(f"✓ Verified: Saved model is A2Transformer in {args.output_dir}", flush=True)
    else:
        print(f"⚠ WARNING: Saved model architecture is {saved_config.get('architectures')}, expected A2Transformer!", flush=True)
else:
    print(f"⚠ WARNING: Could not find config.json at {saved_config_path}", flush=True)
