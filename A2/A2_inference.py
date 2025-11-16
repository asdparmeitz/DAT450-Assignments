import torch
import sys
import os
import nltk
from transformers import PreTrainedModel

# Ensure required NLTK tokenizers are available
nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
A1_DIR = os.path.join(BASE_DIR, 'A1')

# Add A1 to path and import tokenizer
sys.path.insert(0, A1_DIR)
from A1 import A1Tokenizer

# Import the Transformer model
from A2_skeleton import A2Transformer, A2ModelConfig

# Define lowercase_tokenizer function - needed for pickle deserialization
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

# Make it available in __main__ module for pickle
import __main__
if not hasattr(__main__, 'lowercase_tokenizer'):
    __main__.lowercase_tokenizer = lowercase_tokenizer


def load_model_and_tokenizer(model_dir=None, tokenizer_path=None):
    """Load the trained model and tokenizer."""
    if model_dir is None:
        model_dir = os.path.join(SCRIPT_DIR, 'trainer_output')
    if tokenizer_path is None:
        tokenizer_path = os.path.join(A1_DIR, 'tokenizer.pkl')
    
    print(f"Loading model from: {model_dir}")
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    # Load tokenizer
    # The tokenizer was pickled with a reference to lowercase_tokenizer
    # We need to make sure it's available in __main__ module
    import pickle
    
    # First, try the standard method
    try:
        tokenizer = A1Tokenizer.from_file(tokenizer_path)
        print(f"✓ Tokenizer loaded (vocab_size: {len(tokenizer)})")
    except (AttributeError, ModuleNotFoundError) as e:
        print(f"Standard loading failed: {e}")
        print("Trying direct pickle load...")
        # Direct pickle load - the function should be in __main__ now
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"✓ Tokenizer loaded (vocab_size: {len(tokenizer)})")
    
    # Load model
    model = A2Transformer.from_pretrained(model_dir)
    print(f"✓ Model loaded")
    
    return model, tokenizer


def predict_next(tokenizer, model, text, device, k=5):
    """Predict the top-k next tokens given a text prompt.
    
    Args:
        tokenizer: The tokenizer
        model: The trained model
        text: Input text string
        device: torch device
        k: Number of top predictions to return
    
    Returns:
        List of top-k predicted tokens
    """
    model.eval()
    with torch.no_grad():
        enc = tokenizer(text, return_tensors='pt', padding=False, truncation=True)
        ids = enc["input_ids"].to(device)
        X = ids[:, :-1]  # All tokens except the last (EOS)
        logits = model(X)
        last_logits = logits[:, -1, :]  # Get logits for the last position
        topk = torch.topk(last_logits, k=k, dim=-1)
        idxs = topk.indices[0].tolist()
        return [tokenizer.id2word.get(i, tokenizer.unk_token) for i in idxs]


def decode_token_ids(tokenizer, token_ids):
    """Decode a list of token IDs back to text."""
    tokens = [tokenizer.id2word.get(id, tokenizer.unk_token) for id in token_ids]
    # Filter out special tokens for display
    filtered_tokens = [t for t in tokens if t not in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token]]
    return ' '.join(filtered_tokens)


def generate_text(tokenizer, model, prompt, device, max_length=50, temperature=1.0, top_k=50):
    """Generate text autoregressively.
    
    Args:
        tokenizer: The tokenizer
        model: The trained model
        prompt: Starting text
        device: torch device
        max_length: Maximum length of generated sequence
        temperature: Sampling temperature (higher = more random)
        top_k: Only sample from top-k tokens
    
    Returns:
        Generated text string
    """
    model.eval()
    with torch.no_grad():
        # Tokenize the prompt
        enc = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
        input_ids = enc["input_ids"].to(device)
        
        generated_ids = input_ids.clone()
        
        for _ in range(max_length):
            # Get logits for the last token
            logits = model(generated_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if we generate EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        # Convert back to text
        generated_text = decode_token_ids(tokenizer, generated_ids[0].tolist())
        return generated_text


def evaluate_perplexity(tokenizer, model, texts, device, batch_size=32):
    """Evaluate perplexity on a list of texts.
    
    Args:
        tokenizer: The tokenizer
        model: The trained model
        texts: List of text strings
        device: torch device
        batch_size: Batch size for evaluation
    
    Returns:
        Average perplexity
    """
    from torch.utils.data import DataLoader
    from datasets import Dataset
    
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    V = model.config.vocab_size
    
    dataset = Dataset.from_dict({"text": texts})
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in loader:
            batch_texts = batch["text"]
            enc = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            X, Y = ids[:, :-1], ids[:, 1:]
            
            logits = model(X)
            loss = loss_func(logits.reshape(-1, V), Y.reshape(-1))
            
            num_tokens = (Y != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_ce = total_loss / max(1, total_tokens)
    perplexity = float(torch.exp(torch.tensor(avg_ce)))
    
    return perplexity, avg_ce


if __name__ == "__main__":
    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}\n")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    model.to(device)
    model.eval()
    
    print("\n" + "="*60)
    print("INFERENCE EXAMPLES")
    print("="*60 + "\n")
    
    # Example 1: Predict next tokens
    print("1. Top-5 next token predictions:")
    print("-" * 60)
    test_prompts = [
        "She lives in San",
        "The capital of France is",
        "In the beginning",
        "The quick brown",
        "Machine learning is"
    ]
    
    for prompt in test_prompts:
        top5 = predict_next(tokenizer, model, prompt, device, k=5)
        print(f"  '{prompt}' → {top5}")
    
    print("\n" + "="*60)
    print("2. Text generation (autoregressive):")
    print("-" * 60)
    
    # Example 2: Generate text
    generation_prompts = [
        "The weather today is",
        "Once upon a time",
        "Artificial intelligence"
    ]
    
    for prompt in generation_prompts:
        generated = generate_text(
            tokenizer, model, prompt, device, 
            max_length=30, temperature=0.8, top_k=50
        )
        print(f"  Prompt: '{prompt}'")
        print(f"  Generated: {generated}")
        print()
    
    print("="*60)
    print("Inference complete!")

