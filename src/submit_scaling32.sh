#!/usr/bin/env bash
#SBATCH --job-name=om8_pipeline_test_villian
#SBATCH --error=error_%j.err
#SBATCH --output=Active1output_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --partition=cpu
#SBATCH --mem=30GB
#SBATCH --exclusive
# --gres=gpu:h100:3

# Load required modules # -A 156178103677
module purge
module load Anaconda3/2024.02-1
# module load VTune/2025.0.0
# Initialize conda
conda init
source ~/.bashrc

# Activate your conda environment
conda activate openmm_test

set -euo pipefail

# ---- paths ----
BASE="../data_scaling_trajs/"                         # main folder: holds ALL 32 seeds + residues.txt + baseline
SRC_SEEDS="$BASE/uncertain_seeds"
OUT="../test_scaling"                # where active_N_runM roots get created
SCRIPT="scaling.py"

# ---- which active levels, and how many repeats each ----
ACTIVE_LEVELS=(1 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32)
# ACTIVE_LEVELS=(1)
N_REPEATS=1                                 # run1, run2, run3

# all seed folders in BASE, sorted (deterministic pick)
mapfile -t ALL_SEEDS < <(find "$SRC_SEEDS" -mindepth 1 -maxdepth 1 -type d | sort -t_ -k2,2n -k4,4n -k6,6n)

echo "BASE has ${#ALL_SEEDS[@]} seeds"

for A in "${ACTIVE_LEVELS[@]}"; do
  for R in $(seq 1 "$N_REPEATS"); do
    RUN_ROOT="$OUT/active_${A}_run${R}"
    SEEDS_DST="$RUN_ROOT/uncertain_seeds"
    echo "=== active=$A run=$R -> $RUN_ROOT ==="

    if [ -d "$RUN_ROOT" ]; then
      echo "  exists — skipping (delete it to redo)."
      continue
    fi

    mkdir -p "$SEEDS_DST" "$RUN_ROOT/uncertain_descendants"

    # copy exactly A seed folders from BASE
    for i in $(seq 0 $((A-1))); do
      cp -r "${ALL_SEEDS[$i]}" "$SEEDS_DST/"
    done

    # copy the ROOT-level files the script needs
    cp "$BASE/residues.txt" "$RUN_ROOT/"
    [ -f "$BASE/variables_global.pkl" ]           && cp "$BASE/variables_global.pkl" "$RUN_ROOT/"
    cp "$BASE"/variables_[0-3].pkl "$RUN_ROOT/" 2>/dev/null || true

    # clean any stale markers that might be inside copied seeds
    find "$SEEDS_DST" -name ".COMPLETED" -delete
    find "$SEEDS_DST" -name "*.chk"        -delete    # remove checkpoint -> fresh traj_id
    find "$SEEDS_DST" -name "traj[0-32]*.pdb" -delete   # remove working pdb (keep seed .pdb
    PULSE_RUN_ID="active_${A}_run${R}" python "$SCRIPT" \
        --root "$RUN_ROOT" \
        --active "$A" \
        --num_seeds "$A" \
        --fixed_windows 20 \
        --sim_jitter_frac 0.08 \
        --no_descendants
        
  done
done

#nohup ./submit.sh > custom_log.txt 2>&1 &
