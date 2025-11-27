#!/bin/bash
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L40s:1

mkdir -p logs
set -euo pipefail

MODEL_PATH="/data/courses/2025_dat450_dit247/models/OLMo-2-0425-1B"
ADAPTER_PATH=
MAX_LENGTH=512
MAX_NEW_TOKENS=128
TEMPERATURE=1.0

COMMON_ARGS=(
  --model-path "$MODEL_PATH"
  --max-length "$MAX_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
)

source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate
which python3

run_predict() {
  python3 predict.py "${COMMON_ARGS[@]}" "$@"
}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

echo "Adapter path: ${ADAPTER_PATH:-<none>}"

if [ -n "$ADAPTER_PATH" ]; then
  EXTRA_ARGS=(--adapter-path "$ADAPTER_PATH")
else
  EXTRA_ARGS=()
fi

if [ "$#" -gt 0 ]; then
  run_predict "${EXTRA_ARGS[@]}" "$@"
else
  echo "No instruction provided; running sample prompts..."
  SAMPLE_INSTRUCTIONS=(
    "Summarize the following review in one sentence."
    "Rewrite the following text in a more formal and academic style."
    "Decide whether the sentiment of the following review is positive, negative, or mixed. Answer with one word."
    "Answer the question step by step, and then give the final answer on a new line starting with 'Answer:'."
  )
  SAMPLE_INPUTS=(
    "The movie was slow at first, but the acting and soundtrack were incredible."
    "I think this project turned out pretty cool. We had some issues in the middle, but overall the results look solid and I'm happy with what we did."
    "The plot was all over the place and I almost left halfway through, but the last 20 minutes were surprisingly emotional and well-acted."
    "A shop sells notebooks for 25 kronor each. You have 140 kronor. How many notebooks can you buy, and how much money will you have left?"
  )
  for idx in "${!SAMPLE_INSTRUCTIONS[@]}"; do
    echo ""
    echo ">>> Sample $((idx + 1))"
    run_predict "${EXTRA_ARGS[@]}" "${SAMPLE_INSTRUCTIONS[$idx]}" --input "${SAMPLE_INPUTS[$idx]}"
  done
fi

echo "Job finished at: $(date)"
