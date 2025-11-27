#!/bin/bash
#SBATCH --output=logs/fine_tuning_%j.out
#SBATCH --error=logs/fine_tuning_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:L40s:1

mkdir -p logs
set -euo pipefail

DATASET_DIR="/data/courses/2025_dat450_dit247/datasets/alpaca-cleaned"
MODEL_PATH="/data/courses/2025_dat450_dit247/models/OLMo-2-0425-1B"
OUTPUT_DIR="./outputs"
NUM_EPOCHS=2
SEED=101
MAX_TRAIN_SAMPLES=2000
MAX_TEST_SAMPLES=200
MAX_LENGTH=512
MAX_NEW_TOKENS=128
DEVICE="cuda"

TRAIN_ARGS=(
  --dataset-dir "$DATASET_DIR"
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --num-epochs "$NUM_EPOCHS"
  --seed "$SEED"
  --max-train-samples "$MAX_TRAIN_SAMPLES"
  --max-test-samples "$MAX_TEST_SAMPLES"
  --max-length "$MAX_LENGTH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --device "$DEVICE"
)

source /data/courses/2025_dat450_dit247/venvs/dat450_venv/bin/activate
which python3

run_training() {
  python3 main.py "${TRAIN_ARGS[@]}" "$@"
}

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

run_training "$@"
echo "Job finished at: $(date)"
