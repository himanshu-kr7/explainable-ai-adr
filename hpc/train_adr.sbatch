#!/bin/bash
#SBATCH --job-name=adr_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Target ADR: $ADR"
echo "Started: $(date)"

source ~/miniconda3/bin/activate
conda activate mtp_env

nvidia-smi --query-gpu=name --format=csv,noheader

mkdir -p logs Results

python Scripts/Phase2_GNN_Training/train_job.py \
    --target_adr "$ADR" \
    --epochs 50 \
    --save_path "Results/Phase2_TAG_Model_${ADR}.pth"

EXIT_CODE=$?
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
