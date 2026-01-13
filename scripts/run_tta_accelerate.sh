#!/bin/bash -l
#SBATCH -J run_tta
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="logs/run_tta.out"
#SBATCH --error="logs/run_tta.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

# ...existing code for parsing arguments...

# For Option 3 (Accelerate) - recommended
accelerate launch --num_processes=2 --mixed_precision=bf16 \
    src/scripts/tta/run_tta_generation.py +preset="$PRESET"

# OR for Option 2 (Manual multi-GPU) - just run normally
# "$ENV_DIR/bin/python" src/scripts/tta/run_tta_generation.py +preset="$PRESET"