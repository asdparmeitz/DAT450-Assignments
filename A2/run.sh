#!/bin/bash
#SBATCH --job-name=A2_training
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L4:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -p long
#SBATCH --nodelist=callisto

# Activate virtual environment
source ~/A1_project/.venv/bin/activate

# Navigate to project directory
cd ~/A1_project

# Create logs directory if it doesn't exist
mkdir -p A2/logs

# Verify GPU is available
echo "GPU Information:"
nvidia-smi

# CUDA_LAUNCH_BLOCKING disabled for better performance (not needed with single GPU)
# export CUDA_LAUNCH_BLOCKING=1

# Run training script
python A2/A2_train.py

