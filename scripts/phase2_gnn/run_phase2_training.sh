#!/bin/bash
#SBATCH --job-name=MTP_TAG_Full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=Results/phase2_training_output.txt

echo "====================================================="
echo "🚀 Starting MTP Phase 2: Decagon + ClinicalBERT TAG"
echo "Job execution started!"
echo "====================================================="

# Load the specific Param Rudra conda environment
source ~/miniconda3/bin/activate
conda activate mtp_env

# Run the heavy training script with unbuffered output (-u)
python -u Scripts/train_job.py --use_real_features True --full_graph True

echo "====================================================="
echo "✅ Job finished successfully!"
echo "====================================================="