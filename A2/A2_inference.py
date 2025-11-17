import torch
import sys
import os
import nltk
from transformers import PreTrainedModel
from torch.distributions import Categorical

nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
A1_DIR = os.path.join(BASE_DIR, 'A1')

sys.path.insert(0, A1_DIR)
from A1 import A1Tokenizer

from A2_skeleton import A2Transformer, A2ModelConfig

def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

import __main__
if not hasattr(__main__, 'lowercase_tokenizer'):
    __main__.lowercase_tokenizer = lowercase_tokenizer


def load_model_and_tokenizer(model_dir=None, tokenizer_path=None):
    if model_dir is None:
        model_dir = os.path.join(SCRIPT_DIR, 'trainer_output')
        print(f"Using pre-trained model directory: {model_dir}")
    
    if tokenizer_path is None:
        a1_tokenizer = os.path.join(A1_DIR, 'tokenizer.pkl')
        base_tokenizer = os.path.join(BASE_DIR, 'tokenizer.pkl')
        
        if os.path.exists(a1_tokenizer):
            tokenizer_path = a1_tokenizer
        elif os.path.exists(base_tokenizer):
            tokenizer_path = base_tokenizer
        else:
            tokenizer_path = a1_tokenizer
    
    print(f"Loading model from: {model_dir}")
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    import pickle
    tokenizer = None
    
    try:
        tokenizer = A1Tokenizer.from_file(tokenizer_path)
        print(f"Tokenizer loaded (vocab_size: {len(tokenizer)})")
    except (AttributeError, ModuleNotFoundError) as e:
        print(f"Standard loading failed: {e}")
        print("Trying direct pickle load...")
        try:
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            print(f"Tokenizer loaded (vocab_size: {len(tokenizer)})")
        except Exception as e2:
            print(f"Direct pickle load also failed: {e2}")
            if tokenizer_path != a1_tokenizer and os.path.exists(a1_tokenizer):
                print(f"Trying fallback to A1 tokenizer: {a1_tokenizer}")
                try:
                    tokenizer = A1Tokenizer.from_file(a1_tokenizer)
                    print(f"Tokenizer loaded from A1 directory (vocab_size: {len(tokenizer)})")
                except Exception as e3:
                    print(f"Fallback also failed: {e3}")
                    raise
            else:
                raise
    
    try:
        model = A2Transformer.from_pretrained(model_dir)
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    return model, tokenizer


def predict_next_word(model, tokenizer, prompt, device):
    model.eval()
    with torch.no_grad():
        enc = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
        input_ids = enc["input_ids"].to(device)
        
        logits = model(input_ids)
        
        last_pos = input_ids.shape[1] - 1
        if input_ids.shape[1] > 1 and input_ids[0, -1].item() == tokenizer.eos_token_id:
            last_pos -= 1
        last_logits = logits[:, last_pos, :]
        
        predicted_idx = torch.argmax(last_logits, dim=-1).item()
        predicted_word = tokenizer.id2word.get(predicted_idx, tokenizer.unk_token)
        
        return predicted_word


def generate_text(model, tokenizer, prompt, device, max_length=50, temperature=1.0, topk=None):
    model.eval()
    with torch.no_grad():
        enc = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
        input_ids = enc["input_ids"].to(device)
        
        generated_ids = input_ids.clone()
        
        for step in range(max_length):
            logits = model(generated_ids)
            next_token_logits = logits[:, -1, :]
            
            next_token_logits = next_token_logits / temperature
            
            if topk is not None and topk > 0:
                topk_values, topk_indices = torch.topk(next_token_logits, k=min(topk, next_token_logits.shape[-1]), dim=-1)
                
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits.scatter_(-1, topk_indices, topk_values)
                next_token_logits = filtered_logits
            
            dist = Categorical(logits=next_token_logits)
            next_token = dist.sample()
            next_token = next_token.unsqueeze(1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        generated_token_ids = generated_ids[0].tolist()
        tokens = []
        for token_id in generated_token_ids:
            token = tokenizer.id2word.get(token_id, tokenizer.unk_token)
            if token not in [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token]:
                tokens.append(token)
        
        return ' '.join(tokens)


def load_olmo_model(local_dir='/data/courses/2025_dat450_dit247/models/OLMo-2-0425-1B'):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        if not os.path.exists(local_dir):
            print(f"OLMo model directory not found: {local_dir}")
            print("Skipping OLMo model loading.")
            return None, None
        
        print(f"Loading OLMo-2 model from: {local_dir}")
        tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(local_dir, local_files_only=True)
        print("OLMo-2 model loaded")
        return model, tokenizer
    except Exception as e:
        print(f"Could not load OLMo model: {e}")
        return None, None


def generate_text_with_olmo(model, tokenizer, prompt, device, max_length=50, temperature=1.0, topk=None):
    model.eval()
    with torch.no_grad():
        enc = tokenizer(prompt, return_tensors='pt', padding=False, truncation=True)
        input_ids = enc["input_ids"].to(device)
        
        generated_ids = input_ids.clone()
        
        for step in range(max_length):
            output = model(generated_ids)
            if hasattr(output, 'logits'):
                logits = output.logits
            else:
                logits = output
            
            next_token_logits = logits[:, -1, :]
            
            next_token_logits = next_token_logits / temperature
            
            if topk is not None and topk > 0:
                topk_values, topk_indices = torch.topk(next_token_logits, k=min(topk, next_token_logits.shape[-1]), dim=-1)
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits.scatter_(-1, topk_indices, topk_values)
                next_token_logits = filtered_logits
            
            dist = Categorical(logits=next_token_logits)
            next_token = dist.sample()
            next_token = next_token.unsqueeze(1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}\n")
    

    print("LOADING PRE-TRAINED TRANSFORMER MODEL")

    model, tokenizer = load_model_and_tokenizer()
    model.to(device)
    model.eval()
    

    print("1. PREDICTING NEXT WORD")

    
    test_prompts = [
        "he lives in san",
        "The capital of France is",
        "In the beginning",
        "Machine learning is"
    ]
    
    for prompt in test_prompts:
        predicted_word = predict_next_word(model, tokenizer, prompt, device)
        print(f"  '{prompt}' {predicted_word}")
    

    print("GENERATING TEXTS")

    
    generation_prompts = [
        'In natural language processing, a Transformer',
        'Is Stockholm the capital of Sweden? Answer yes or no. The answer is',
        'Write a Python program that reverses a list.'
    ]
    
    print("\n--- Default parameters (temperature=1.0, topk=None) ---")
    for prompt in generation_prompts:
        generated = generate_text(model, tokenizer, prompt, device, max_length=50, temperature=1.0, topk=None)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")
    
    print("\n--- With temperature=0.8, topk=50 ---")
    for prompt in generation_prompts:
        generated = generate_text(model, tokenizer, prompt, device, max_length=50, temperature=0.8, topk=50)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")
    
    print("\n--- With temperature=1.5, topk=100 ---")
    for prompt in generation_prompts:
        generated = generate_text(model, tokenizer, prompt, device, max_length=50, temperature=1.5, topk=100)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: {generated}")
    

    print("3. COMPARING TO PRE-TRAINED OLMo-2 MODEL")

    
    olmo_model, olmo_tokenizer = load_olmo_model()
    
    if olmo_model is not None and olmo_tokenizer is not None:
        olmo_model.to(device)
        olmo_model.eval()
        
        print("\n--- OLMo-2 predictions (temperature=0.8, topk=50) ---")
        for prompt in generation_prompts:
            generated = generate_text_with_olmo(olmo_model, olmo_tokenizer, prompt, device, 
                                                max_length=50, temperature=0.8, topk=50)
            print(f"\nPrompt: '{prompt}'")
            print(f"Generated: {generated}")
    else:
        print("\nOLMo-2 model not available. Skipping comparison.")
    

    print("Inference complete!")

