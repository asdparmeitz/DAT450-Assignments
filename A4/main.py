import argparse
import json
import os
import random
import time

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from .data_utils import build_prompt, create_data_collator, tokenize_helper
from .lora import make_lora_model
from .utils import (
    RougeMetricComputer,
    compare_models_on_examples,
    create_stratification_label,
    make_trainer,
    num_trainable_parameters,
    print_results_table,
)

# Global config
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Defaults for CLI overrides
DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_DATASET_DIR = "/data/courses/2025_dat450_dit247/datasets/alpaca-cleaned"
DEFAULT_MODEL_NAME_OR_PATH = "/data/courses/2025_dat450_dit247/models/OLMo-2-0425-1B"
DEFAULT_NUM_EPOCHS = 2
DEFAULT_SEED = 101
DEFAULT_MAX_TRAIN_SAMPLES = 2000
DEFAULT_MAX_TEST_SAMPLES = 200
DEFAULT_MAX_LENGTH = 512
DEFAULT_MAX_NEW_TOKENS = 128


def parse_args():
    parser = argparse.ArgumentParser(description="Student version of the Alpaca SFT training script.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_NAME_OR_PATH)
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-train-samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES)
    parser.add_argument("--max-test-samples", type=int, default=DEFAULT_MAX_TEST_SAMPLES)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use (default auto-detect).",
    )
    return parser.parse_args()


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


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = vars(args)
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    for k, v in config.items():
        print(f"{k}: {v}")
    print("=" * 80)

    # 1. Load and Prepare Alpaca Dataset
    print("\n" + "=" * 80)
    print("LOADING ALPACA DATASET")
    print("=" * 80)

    alpaca_dataset = load_dataset(args.dataset_dir)

    # Filter out empty outputs from the train split
    alpaca_dataset["train"] = alpaca_dataset["train"].filter(lambda x: x["output"] is not None and x["output"].strip() != "")

    print("\nALPACA DATASET:")
    print(alpaca_dataset)

    # Create stratified train/test split
    ds = alpaca_dataset["train"].map(lambda x: create_stratification_label(x, columns_to_check=["input"]))

    # Turn strat_label into a ClassLabel so we can stratify
    ds = ds.class_encode_column("strat_label")

    ds = (
        ds.shuffle(seed=args.seed)
        .select(range(args.max_train_samples + args.max_test_samples))
        .train_test_split(
            train_size=args.max_train_samples,
            test_size=args.max_test_samples,
            stratify_by_column="strat_label",
            seed=args.seed,
        )
    )

    print("\nALPACA + SUBSAMPLE:")
    print(ds)

    # 2. Build Alpaca-style Prompts
    print("\n" + "=" * 80)
    print("BUILDING PROMPTS")
    print("=" * 80)

    ds_sft = ds.map(lambda x: build_prompt(x, PROMPT_NO_INPUT, PROMPT_WITH_INPUT))
    print("\nSample with prompt:")
    print(json.dumps(ds_sft["train"][0], indent=2))

    # 3. Tokenization
    print("\n" + "=" * 80)
    print("TOKENIZATION")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    tokenized_ds_sft = DatasetDict(
        {
            "train": ds_sft["train"].map(lambda x: tokenize_helper(x, tokenizer, args.max_length)),
            "test": ds_sft["test"].map(lambda x: tokenize_helper(x, tokenizer, args.max_length)),
        }
    )

    print("\nTOKENIZED DATASET:")
    print(tokenized_ds_sft)

    # Create data collator and metrics
    data_collator = create_data_collator(tokenizer)
    compute_metrics = RougeMetricComputer(tokenizer)

    # Evaluate Pretrained Model (No Fine-tuning)
    print("\n" + "=" * 80)
    print("EVALUATING PRETRAINED MODEL")
    print("=" * 80)

    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)

    pretrained_eval_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "pretrained"),
        eval_strategy="no",
        per_device_eval_batch_size=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        batch_eval_metrics=True,
        eval_accumulation_steps=1,
    )

    pretrained_trainer = make_trainer(
        pretrained_model,
        pretrained_eval_args,
        tokenized_ds_sft,
        compute_metrics,
        data_collator,
    )

    t0 = time.perf_counter()
    pretrained_eval_metrics = pretrained_trainer.evaluate()
    pretrained_eval_time = time.perf_counter() - t0

    pretrained_eval_loss = float(pretrained_eval_metrics["eval_loss"])
    pretrained_rougeL = pretrained_eval_metrics.get("eval_rougeL", None)

    print("\nPRETRAINED EVAL METRICS:")
    print(json.dumps(pretrained_eval_metrics, indent=2))

    # Train Baseline Model (Full SFT)
    print("\n" + "=" * 80)
    print("TRAINING BASELINE MODEL (FULL SFT)")
    print("=" * 80)

    baseline_training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "trainer_sft_baseline"),
        eval_strategy="epoch",
        logging_steps=2000,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
        batch_eval_metrics=True,
        eval_accumulation_steps=1,
    )

    base_model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)
    print(f"Full SFT trainable params: {num_trainable_parameters(base_model)}")

    baseline_trainer = make_trainer(base_model, baseline_training_args, tokenized_ds_sft, compute_metrics, data_collator)

    t0 = time.perf_counter()
    baseline_trainer.train()
    baseline_train_time = time.perf_counter() - t0

    # Save model
    baseline_model_dir = os.path.join(args.output_dir, "trainer_sft_baseline", "finetuned_sft_baseline.model")
    baseline_trainer.save_model(baseline_model_dir)
    tokenizer.save_pretrained(baseline_model_dir)

    t0 = time.perf_counter()
    baseline_eval_metrics = baseline_trainer.evaluate()
    baseline_eval_time = time.perf_counter() - t0

    baseline_eval_loss = float(baseline_eval_metrics["eval_loss"])
    baseline_rougeL = baseline_eval_metrics.get("eval_rougeL", None)

    print("\nBASELINE EVAL METRICS:")
    print(json.dumps(baseline_eval_metrics, indent=2))

    # Train LoRA Model
    print("\n" + "=" * 80)
    print("TRAINING LORA MODEL")
    print("=" * 80)

    lora_model, lora_train_time, lora_eval_metrics, lora_eval_time = make_lora_model(
        rank=8,
        name="lora8_sft",
        n_epochs=args.num_epochs,
        model_name_or_path=args.model_path,
        device=args.device,
        output_dir=args.output_dir,
        tokenized_ds_sft=tokenized_ds_sft,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    lora_eval_loss = float(lora_eval_metrics["eval_loss"])
    lora_rougeL = lora_eval_metrics.get("eval_rougeL", None)

    print("\nLORA EVAL METRICS:")
    print(json.dumps(lora_eval_metrics, indent=2))

    # Summary Results
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)

    results = [
        {
            "name": "Pretrained",
            "trainable_params": "N/A",
            "train_time": "N/A",
            "eval_time": f"{pretrained_eval_time:.2f}s",
            "eval_loss": f"{pretrained_eval_loss:.4f}",
            "rougeL": f"{pretrained_rougeL:.4f}" if pretrained_rougeL else "N/A",
        },
        {
            "name": "Baseline SFT",
            "trainable_params": num_trainable_parameters(base_model),
            "train_time": f"{baseline_train_time:.2f}s",
            "eval_time": f"{baseline_eval_time:.2f}s",
            "eval_loss": f"{baseline_eval_loss:.4f}",
            "rougeL": f"{baseline_rougeL:.4f}" if baseline_rougeL else "N/A",
        },
        {
            "name": "LoRA-8 SFT",
            "trainable_params": num_trainable_parameters(lora_model),
            "train_time": f"{lora_train_time:.2f}s",
            "eval_time": f"{lora_eval_time:.2f}s",
            "eval_loss": f"{lora_eval_loss:.4f}",
            "rougeL": f"{lora_rougeL:.4f}" if lora_rougeL else "N/A",
        },
    ]

    print_results_table(results)

    # Qualitative Comparison
    models_to_compare = {
        "pretrained": pretrained_model,
        "baseline_sft": base_model,
        "lora_sft": lora_model,
    }

    test_cases = [
        None,  # Placeholder for test set example
        (
            "Summarize the following review in one sentence.",
            "The movie was slow at first, but the acting and soundtrack were incredible.",
            "",
        ),
        (
            "Rewrite the following text in a more formal and academic style.",
            "I think this project turned out pretty cool. We had some issues in the middle, but overall the results look solid and I'm happy with what we did.",
            "",
        ),
        (
            "Decide whether the sentiment of the following review is positive, negative, or mixed. Answer with one word.",
            "The plot was all over the place and I almost left halfway through, but the last 20 minutes were surprisingly emotional and well-acted.",
            "",
        ),
        (
            "Answer the question step by step, and then give the final answer on a new line starting with 'Answer:'.",
            "A shop sells notebooks for 25 kronor each. You have 140 kronor. How many notebooks can you buy, and how much money will you have left?",
            "",
        ),
    ]
    s = ds_sft["test"][0]
    test_cases[0] = (s["instruction"], s["input"], s["output"])

    compare_models_on_examples(
        test_cases=test_cases,
        models_dict=models_to_compare,
        tokenizer=tokenizer,
        temperature=1.0,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        prompt_no_input=PROMPT_NO_INPUT,
        prompt_with_input=PROMPT_WITH_INPUT,
    )


if __name__ == "__main__":
    main()
