#!/bin/bash
#SBATCH --job-name=om8_pipeline_test_villian
#SBATCH --error=error_%j.err
#SBATCH --output=NEWtrajsoutput_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --mem=30GB
#SBATCH --gres=gpu:h100:2

# Load required modules # -A 156178103677
module purge
module load Anaconda3/2024.02-1
# module load VTune/2025.0.0
# Initialize conda
conda init
source ~/.bashrc

# Activate your conda environment
conda activate openmm_test


python NKB.py --root NKBA/  --total_frames 10000 --num_seeds 10 --active 2 --frames_per_window 50 