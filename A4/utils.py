import evaluate
import torch
from transformers import Trainer

from data_utils import build_prompt


def print_results_table(results):
    """Pretty-print a list of result dictionaries as a simple table."""

    cols = [
        "name",
        "trainable_params",
        "train_time",
        "eval_time",
        "eval_loss",
        "rougeL",
    ]
    col_widths = {
        c: max(len(c), max((len(str(r.get(c, ""))) for r in results), default=0))
        for c in cols
    }

    def fmt_row(row):
        return " | ".join(str(row.get(c, "")).ljust(col_widths[c]) for c in cols)

    print(fmt_row({c: c for c in cols}))
    print("-" * (sum(col_widths.values()) + 3 * (len(cols) - 1)))
    for r in results:
        print(fmt_row(r))


def create_stratification_label(example, columns_to_check=["input", "output"]):
    """
    Create a composite stratification label based on which fields are empty/non-empty.

    For example, "1_0" means:
        - input is non-empty
        - output is empty
    """
    label_parts = []
    for col in columns_to_check:
        label_parts.append("1" if example.get(col, "").strip() else "0")

    return {"strat_label": "_".join(label_parts)}


def num_trainable_parameters(model):
    """Count number of trainable parameters (requires_grad=True)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_trainer(
    model, training_args, tokenized_ds_sft, compute_metrics, data_collator
):
    """Create a Trainer for SFT on tokenized_ds_sft."""

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds_sft["train"],
        eval_dataset=tokenized_ds_sft["test"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    return trainer


class RougeMetricComputer:
    """
    Stateful metric for batch_eval_metrics=True.

    It:
      - accumulates predictions and references across batches
      - computes ROUGE-L once at the end (compute_result=True)
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge = evaluate.load("rouge")
        self.all_predictions = []
        self.all_references = []

    def __call__(self, eval_pred, compute_result=False):
        """Accumulate predictions and compute at the end."""

        logits, labels = eval_pred
        pred_ids = logits.argmax(axis=-1)

        # Collect decoded answer-span text from each example in the batch
        for p, lbl in zip(pred_ids, labels):
            mask = lbl != -100
            if mask.sum() == 0:
                continue

            ref_ids = lbl[mask]
            pred_ids_filtered = p[mask]

            ref_text = self.tokenizer.decode(ref_ids, skip_special_tokens=True)
            pred_text = self.tokenizer.decode(
                pred_ids_filtered, skip_special_tokens=True
            )

            self.all_references.append(ref_text.strip())
            self.all_predictions.append(pred_text.strip())

        # Only compute at the very end of eval
        if compute_result:
            if len(self.all_references) > 0:
                scores = self.rouge.compute(
                    predictions=self.all_predictions,
                    references=self.all_references,
                )

                # Clear accumulated data for next eval call
                self.all_predictions = []
                self.all_references = []
                return {"rougeL": scores["rougeL"]}
            else:
                return {}
        else:
            return {}


def build_prompt_from_instruction(instruction, inp, prompt_no_input, prompt_with_input):
    """
    Build the prompt string from an instruction and optional input by
    delegating to the shared `build_prompt` helper.

    Args:
        instruction: The task description.
        inp: Optional input/context string.
    """
    example = {"instruction": instruction, "input": inp or "", "output": ""}
    prompt_bundle = build_prompt(example, prompt_no_input, prompt_with_input)
    return prompt_bundle["prompt"]


def generate_response(
    model,
    tokenizer,
    instruction,
    inp="",
    max_length=512,
    max_new_tokens=128,
    temperature=0.0,
    prompt_no_input="",
    prompt_with_input="",
):
    """
    Generate a response for a given instruction (+ optional input).

    Args:
        model: Causal LM (baseline SFT, LoRA, etc.).
        tokenizer: Corresponding tokenizer.
        instruction: Instruction string.
        inp: Optional input string.
        max_new_tokens: Max tokens to generate.
        temperature: 0.0 = greedy, >0.0 = sampling.
    """
    model.eval()
    device = next(model.parameters()).device

    prompt = build_prompt_from_instruction(
        instruction, inp, prompt_no_input, prompt_with_input
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    # Only decode the new tokens beyond the prompt
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def compare_models_on_examples(
    test_cases,
    models_dict,
    tokenizer,
    temperature=0.0,
    max_length=512,
    max_new_tokens=128,
    prompt_no_input="",
    prompt_with_input="",
):
    """
    Compare multiple models on a list of test cases.

    Args:
        test_cases: List of tuples (instruction, input, expected_output)
        models_dict: Dict mapping model names to model objects
        tokenizer: Tokenizer to use
        temperature: Sampling temperature (0.0 = greedy)
        max_length: Max input length
        max_new_tokens: Max tokens to generate
        prompt_no_input: Template for prompts without input
        prompt_with_input: Template for prompts with input
    """
    print("\n" + "=" * 80)
    print("QUALITATIVE COMPARISON")
    print("=" * 80)

    for idx, (instruction, user_input, gold_output) in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"EXAMPLE {idx}/{len(test_cases)}")
        print(f"{'=' * 80}")
        print(f"\nInstruction: {instruction}")
        print(f"Input: {user_input or '<NO INPUT>'}")

        # Generate predictions for each model
        for model_name, model in models_dict.items():
            print(f"\n{'-' * 30} {model_name.upper()} {'-' * 30}")
            try:
                pred = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    instruction=instruction,
                    inp=user_input or "",
                    max_length=max_length,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    prompt_no_input=prompt_no_input,
                    prompt_with_input=prompt_with_input,
                )
                print(pred or "<NO OUTPUT>")
            except Exception as e:
                print(f"<ERROR: {str(e)}>")

        # Show gold output if available
        if gold_output:
            print(f"\n{'-' * 30} GOLD OUTPUT {'-' * 30}")
            print(gold_output)

        print()  # Extra newline for readability
