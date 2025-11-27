import copy

import torch

# -----------------------------------------------------------------------------
# STUDENT TODOs: Prompt + Tokenization utilities
# -----------------------------------------------------------------------------
# Fill in the functions below.
#   * Turn raw Alpaca examples into prompts (`build_prompt`)
#   * Convert prompt/answer pairs into causal-LM tensors (`tokenize_helper`)
#   * Batch variable-length tensors safely during training (`create_data_collator`)
#
# Hints are provided inline. You can look at the instructor utilities for
# reference after attempting the implementation on your own.
# -----------------------------------------------------------------------------


def build_prompt(example, prompt_no_input, prompt_with_input):
    """
    Build an Alpaca-style prompt and keep the gold output as 'answer'.

    Args:
        example: Dict with keys like 'instruction', 'input', 'output'.
        prompt_no_input: Template for instructions without extra context.
        prompt_with_input: Template for instructions that provide an input.
    Example:
        >>> build_prompt(
        ...     {"instruction": "Do X", "input": "Extra info", "output": "Result"},
        ...     prompt_no_input="... {instruction} ...",
        ...     prompt_with_input="... {instruction} ... {input} ...",
        ... )
        {"prompt": "... Do X ... Extra info ...", "answer": "Result"}
    Returns:
        Dict with the text prompt under "prompt" and the gold answer under "answer".
    """
    instruction = example["instruction"]
    inp = example.get("input", "") or ""
    output = example["output"]

    if inp:
        return {"prompt": prompt_with_input.format(instruction=instruction, input=inp), "answer": output}
    else:
        return {"prompt": prompt_no_input.format(instruction=instruction), "answer": output}
        


def tokenize_helper(batch, tokenizer, max_length):
    """
    Tokenize prompt+answer pairs for causal LM fine-tuning.

    Requirements:
      * Concatenate prompt IDs with answer IDs (answer prefixed with a space).
      *  to `max_length`.
      * Build labels that ignore prompt tokens (set them to -100) and supervise answer tokens.

    Example:
        >>> tokenize_helper(
        ...     {"prompt": "Prompt", "answer": "Answer"},
        ...     tokenizer=my_tokenizer,
        ...     max_length=32,
        ... )
        {"input_ids": [...], "attention_mask": [...], "labels": [...]}

    Returns:
      dict(input_ids=list[int], attention_mask=list[int], labels=list[int])
    """
    prompt = batch["prompt"]
    answer = batch["answer"]
    # Tokenize the propmt and answer separately
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]

    #   2. Concatenate, truncate, and create an attention mask of 1s.
    prompt_len = len(prompt_ids)
    full_ids = prompt_ids + answer_ids
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
    
    # Create an attention mask of 1s
    attention_mask = [1] * len(full_ids)

    # Build labels using -100 for the prompt span and answer token IDs afterward.
    if len(full_ids) <= prompt_len:
        # All tokens are from prompt (prompt was too long, answer was cut off)
        labels = [-100] * len(full_ids)
    else:
        labels = [-100] * prompt_len  # Set the prompt tokens to -100
        
        answer_tokens = full_ids[prompt_len:]
        labels = labels + answer_tokens
    return {"input_ids":full_ids, "attention_mask":attention_mask, "labels":labels}

def create_data_collator(tokenizer):
    """
    Create a custom collate function for causal language modeling.

    The collator should dynamically right-pad:
      * input_ids with tokenizer.pad_token_id
      * attention_mask with 0
      * labels with -100
    to the maximum sequence length within each batch.

    Example:
        >>> collate = create_data_collator(tokenizer)
        >>> collate([{"input_ids": [1], "attention_mask": [1], "labels": [5]}])
        {"input_ids": tensor(...), "attention_mask": tensor(...), "labels": tensor(...)}

    """
    # TODO[student]:
    #   1. Inside `data_collator(batch)`, convert each list of ids into torch tensors.
    #   2. Compute the max length in the batch.
    #   3. Pad every tensor on the right to that length (use torch.full for padding values).
    #   4. Stack the padded tensors into a batch dict and return it.
    #
    # Suggested structure:
    #   def data_collator(batch):
    #       ...
    #       return {
    #           "input_ids": ...,
    #           "attention_mask": ...,
    #           "labels": ...,
    #       }
    #   return data_collator

    def data_collator(batch):
        input_ids_list = [torch.tensor(example["input_ids"], dtype=torch.long) for example in batch]
        attention_masks_list = [torch.tensor(example["attention_mask"], dtype=torch.long) for example in batch]
        labels_list = [torch.tensor(example["labels"], dtype=torch.long) for example in batch]

        # Find max length in this batch
        max_len = "NotImplementedError"

        # Helper pad function: right-pad to max_len
        def pad_to_max(x_list, pad_value):
            # ...
            pass

        # Use tokenizer.pad_token_id for inputs, 0 for attention_mask, -100 for labels
        pad_id = "NotImplementedError"

        batch_input_ids = pad_to_max(input_ids_list, pad_value=pad_id)
        batch_attention_mask = pad_to_max(attention_masks_list, pad_value=0)
        batch_labels = pad_to_max(labels_list, pad_value=-100)

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }
        return batch

    raise NotImplementedError("Implement the causal-LM data collator.")
