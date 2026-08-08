#!/bin/bash


for ACTIVE in {1..8}; do
    SBATCH_FILE="active${ACTIVE}.slurm"
    cat > "$SBATCH_FILE" <<SBATCH
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=123345
#SBATCH --gres=gpu:h100:${ACTIVE}
#SBATCH --job-name=active${ACTIVE}
#SBATCH --error=error_%j.err
#SBATCH --output=Active${ACTIVE}output_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --mem=30GB


module purge
module load Anaconda3/2024.02-1

source ~/.bashrc
conda activate openmm

BASE="../"
TEMPLATE="\${BASE}/sample_data"

RUN_ID=1

RUN_DIR="\${BASE}/Results_pulse_metrics/active${ACTIVE}_\${RUN_ID}"
cp -r "\${TEMPLATE}" "\${RUN_DIR}"

python driver_unceratinity.py \
    --root "\${RUN_DIR}/" \
    --active ${ACTIVE}

echo "[done] active=\$A run complete: \$RUN_DIR"
SBATCH
    # submit it
    JOBID=$(sbatch --parsable "$SBATCH_FILE")
    echo "  submitted job $JOBID  (sbatch: $SBATCH_FILE)"
done

echo ""
echo "All jobs submitted. Watch with:  squeue -u \$USER"
