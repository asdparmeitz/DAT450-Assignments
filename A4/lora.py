import os
import time

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
)

from utils import make_trainer, num_trainable_parameters

# -----------------------------------------------------------------------------
# STUDENT TODOs
# -----------------------------------------------------------------------------
# This file is intentionally left half-finished so you can implement LoRA.
# Complete the following pieces (search for TODO[student]):
#   1. Implement the LoRA adapter parameters and forward pass.
#   2. Decide which linear layers should receive LoRA adapters.
#   3. Wrap the selected layers with your LoRA modules before training.
# Each TODO includes hints. Feel free to add helper functions if needed.
# -----------------------------------------------------------------------------


class LoRA(nn.Module):
    """
    LoRA wrapper around a Linear layer.
    y = W(x) + (alpha/r) * B(A(x))
    """

    def __init__(self, pretrained, rank=6, alpha=12):
        super().__init__()

        # Always keep a reference to the frozen pretrained weight matrix.
        self.pretrained = pretrained

        # [student]: Initialize the low-rank adapter matrices A and B.
        #   * Inspect `pretrained.weight.shape` to find the input and output dims.
        #   * Create `self.A` (shape: in_dim -> rank) and `self.B` (rank -> out_dim).
        #   * Initialize A with a small normal distribution and B with zeros.
        #   * Store the scaling factor alpha / rank in `self.scaling`.
        # Remove the line below once your implementation is ready.
        in_dim, out_dim = pretrained.weight.shape[1], pretrained.weight.shape[0]
        
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        nn.init.normal_(self.A, mean=0.0, std=0.02)
        
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        # [student]: Implement the LoRA forward pass.
        #   * Compute the frozen projection using `self.pretrained(x)`.
        #   * Add the low-rank update `self.B(self.A(x)) * self.scaling`.
        #   * Return the combined result.
        frozen_output = self.pretrained(x)
        
        lora_output = torch.matmul(torch.matmul(x, self.A), self.B) * self.scaling
        return frozen_output + lora_output


def extract_lora_targets(model):
    """
    Decide which Linear sub-modules to wrap with LoRA.

    HINT:
      * Iterate over `model.named_modules()`.
      * Filter modules that are instances of `nn.Linear`.
      * Keep only the ones whose fully-qualified name contains attention clues
        such as 'q_proj', 'k_proj', 'v_proj', or 'o_proj'.
      * Return a dict {qualified_name: module}.
    """
    # [student]: populate the dictionary with eligible layers.

    target_layers = {}

    for name, module in model.named_modules():

        if isinstance(module, nn.Linear):

            if any(proj in name for proj in ['q_proj', 'v_proj']):
                target_layers[name] = module

    return target_layers


def replace_layers(model, named_layers):
    """
    Replace submodules in `model` by name.
    """
    for name, layer in named_layers.items():
        components = name.split(".")
        submodule = model
        for comp in components[:-1]:
            submodule = getattr(submodule, comp)
        setattr(submodule, components[-1], layer)
    return model


def make_lora_model(
    rank,
    name,
    n_epochs,
    model_name_or_path,
    device,
    output_dir,
    tokenized_ds_sft,
    compute_metrics,
    data_collator,
):
    """
    Create, train and save a LoRA SFT model

    Args:
        rank: LoRA rank r.
        name: used for naming the saved model directory.
        n_epochs: number of training epochs.
    """
    # 1) Load fresh base causal LM
    base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

    # 2) Freeze all base parameters
    for p in base_model.parameters():
        p.requires_grad = False

    # 3) Extract all attention-related Linear layers
    original_layers = extract_lora_targets(base_model)
    print(f"Number of LoRA target layers: {len(original_layers)}")

    # 4) Wrap each with LoRA
    wrapped = {}
    for layer_name, layer in original_layers.items():
        wrapped[layer_name] = LoRA(layer, rank=rank, alpha=2 * rank)

    # 5) Replace in the model
    lora_model = replace_layers(base_model, wrapped)

    print(f"Trainable params (LoRA): {num_trainable_parameters(lora_model)}")

    # 6) TrainingArguments for LoRA run
    lora_training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"trainer_lora_{name}"),
        eval_strategy="epoch",
        logging_steps=2000,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        batch_eval_metrics=True,
        eval_accumulation_steps=1,
    )

    # 7) Trainer
    lora_trainer = make_trainer(lora_model, lora_training_args, tokenized_ds_sft, compute_metrics, data_collator)

    # Track training time
    t0 = time.perf_counter()
    lora_trainer.train()
    lora_train_time = time.perf_counter() - t0

    adapter_path = os.path.join(
        output_dir,
        f"trainer_lora_{name}",
        f"finetuned_sft_lora_{name}.model",
        "lora_state_dict.pt",
    )
    os.makedirs(os.path.dirname(adapter_path), exist_ok=True)
    adapter_state = {
        "rank": rank,
        "alpha": 2 * rank,
        "base_model": model_name_or_path,
        "state_dict": {k: v.cpu() for k, v in lora_model.state_dict().items()},
    }
    torch.save(adapter_state, adapter_path)
    print(f"Saved LoRA model to {os.path.join(adapter_path)}")

    # Evaluate LoRA model (teacher-forced loss + ROUGE-L)
    t0 = time.perf_counter()
    lora_eval_metrics = lora_trainer.evaluate()
    lora_eval_time = time.perf_counter() - t0

    return lora_model, lora_train_time, lora_eval_metrics, lora_eval_time
