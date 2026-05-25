#!/bin/bash
#SBATCH --job-name=adr_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

echo "Started: $(date)"
source ~/miniconda3/bin/activate
conda activate mtp_env
nvidia-smi --query-gpu=name --format=csv,noheader

python Scripts/Phase2_GNN_Training/train_job.py \
    --target_adr thrombocytopenia \
    --epochs 1 \
    --save_path Results/test_sanity.pth

echo "Exit code: $?"
echo "Finished: $(date)"
