#!/bin/bash
#==============================================================================
# SLURM Batch Script: Target Contamination & Leakage Diagnostic
# Description: Executes target-blinded assessments to verify zero data leakage.
# Target Architecture: Param Rudra (or generic SLURM HPC Cluster)
#==============================================================================

#SBATCH --job-name=mtp_leak_test        # Job name
#SBATCH --partition=gpu                 # GPU partition
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --cpus-per-task=4               # Request 4 CPU cores
#SBATCH --mem=32G                       # RAM allocation for 5GB graph processing
#SBATCH --time=00:10:00                 # Maximum execution time (HH:MM:SS)
#SBATCH --output=leak_test_%j.out       # Standard output log (%j = SLURM Job ID)
#SBATCH --error=leak_test_%j.err        # Standard error log

# Instantly abort the script if any command fails
set -e

echo "========================================================================"
echo "[START] Initiating Contamination Diagnostic Job"
echo "Date: $(date)"
echo "Compute Node Allocated: $(hostname)"
echo "========================================================================"

# 1. Environment Initialization
echo "[INFO] Initializing Conda Environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mtp_env

# 2. Hardware Diagnostics
echo "[INFO] GPU Hardware Diagnostics:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 3. Job Execution
echo "[INFO] Launching Target Leakage Diagnostic Protocol..."
# The -u flag ensures unbuffered output so logs update in real-time
python -u test_contamination.py

echo "========================================================================"
echo "[SUCCESS] Job successfully completed at $(date)"
echo "========================================================================"