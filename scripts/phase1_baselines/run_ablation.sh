#!/bin/bash
#==============================================================================
# SLURM Batch Script: Heterogeneous GNN Edge-Occlusion Explainer
# Description: Deploys the ablation study on a dedicated GPU compute node.
# Target Architecture: Param Rudra (or generic SLURM HPC Cluster)
#==============================================================================

#SBATCH --job-name=MTP_Explain          # Job name
#SBATCH --partition=gpu                 # GPU partition
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=4               # Request 4 CPU cores for data loaders
#SBATCH --mem=32G                       # RAM allocation for 5GB graph processing
#SBATCH --time=02:00:00                 # Maximum execution time (HH:MM:SS)
#SBATCH --output=explain_job_%j.out     # Standard output log (%j = SLURM Job ID)
#SBATCH --error=explain_job_%j.err      # Standard error log

# Instantly abort the script if any command fails
set -e

echo "========================================================================"
echo "[START] Initiating MTP Explainability Job"
echo "Date: $(date)"
echo "Compute Node Allocated: $(hostname)"
echo "========================================================================"

# 1. Environment Initialization
echo "[INFO] Initializing Conda Environment..."
# (Note for GitHub users: Adjust this path to match your local Anaconda/Miniconda installation)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mtp_env

# 2. Hardware Diagnostics
echo "[INFO] GPU Hardware Diagnostics:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 3. Job Execution
echo "[INFO] Launching Edge-Occlusion Ablation Protocol..."
python explain_biology.py --top_k 5

echo "========================================================================"
echo "[SUCCESS] Job successfully completed at $(date)"
echo "========================================================================"