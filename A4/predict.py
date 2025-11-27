import argparse
import os
import warnings

import torch
from lora import LoRA, extract_lora_targets, replace_layers
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import generate_response

warnings.simplefilter("ignore")

PROMPT_NO_INPUT = """
Below is an instruction that describes a task.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
""".strip()

PROMPT_WITH_INPUT = """
Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
""".strip()

DEFAULT_MODEL = "/data/courses/2025_dat450_dit247/models/OLMo-2-0425-1B"


def parse_args():
    parser = argparse.ArgumentParser(description="Run text-generation inference without training.")
    parser.add_argument("instruction", nargs="?", help="Instruction/prompt to feed the model")
    parser.add_argument("--input", dest="user_input", default="", help="Optional input/context string")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_NAME_OR_PATH", DEFAULT_MODEL))
    parser.add_argument(
        "--adapter-path",
        default=os.environ.get("LORA_ADAPTER_PATH", ""),
        help="Optional path to a saved LoRA adapter checkpoint.",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Max encoded prompt length")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use (defaults to CUDA if available, otherwise CPU).",
    )
    return parser.parse_args()


def build_lora_model_with_config(rank, alpha, base_model_name_or_path, device):
    """Recreate a LoRA-wrapped version of the base model using stored hyperparameters."""

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path).to(device)
    for param in base_model.parameters():
        param.requires_grad = False

    targets = extract_lora_targets(base_model)
    wrapped = {name: LoRA(layer, rank=rank, alpha=alpha) for name, layer in targets.items()}
    return replace_layers(base_model, wrapped)


def load_lora_model(adapter_path, fallback_model_path, device):
    """Load a LoRA adapter checkpoint and rebuild the wrapped model."""

    if not os.path.isfile(adapter_path):
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_path}")

    checkpoint = torch.load(adapter_path, map_location="cpu")
    rank = checkpoint.get("rank") or checkpoint.get("lora_rank")
    alpha = checkpoint.get("alpha") or checkpoint.get("lora_alpha")
    base_model_name_or_path = checkpoint.get("base_model") or checkpoint.get("base_model_name_or_path") or fallback_model_path
    if rank is None or alpha is None:
        raise ValueError("Adapter checkpoint is missing 'rank' or 'alpha' fields.")

    lora_model = build_lora_model_with_config(rank, alpha, base_model_name_or_path, device)
    state_dict = checkpoint["state_dict"]
    missing, unexpected = lora_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[predict] Missing keys when loading adapter: {missing}")
    if unexpected:
        print(f"[predict] Unexpected keys when loading adapter: {unexpected}")

    lora_model.to(device)
    lora_model.eval()
    return lora_model, base_model_name_or_path


def main():
    args = parse_args()
    if args.instruction is None:
        raise SystemExit("Please provide an instruction (or use predict.sh for sample runs).")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    if args.adapter_path:
        model, tokenizer_source = load_lora_model(
            adapter_path=args.adapter_path,
            fallback_model_path=args.model_path,
            device=device,
        )
    else:
        tokenizer_source = args.model_path
        model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        instruction=args.instruction,
        inp=args.user_input,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        prompt_no_input=PROMPT_NO_INPUT,
        prompt_with_input=PROMPT_WITH_INPUT,
    )

    print("\n=== MODEL RESPONSE ===")
    print(f"\nInstruction: {args.instruction}")
    print(f"Input: {args.user_input or '<NO INPUT>'}\n")
    print(response or "<NO OUTPUT>")


if __name__ == "__main__":
    main()
