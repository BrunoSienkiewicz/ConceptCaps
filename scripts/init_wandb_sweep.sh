#!/bin/bash
# Initialize WandB sweep and get sweep ID
# Usage: ./scripts/init_wandb_sweep.sh

cd "$(dirname "$0")/.." || exit 1

# Activate conda environment
CONDA_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz/.conda"
ENV_NAME=$(grep -E '^name:' environment.yml | awk '{print $2}')
conda activate "$ENV_NAME"

# Set WandB directory
ROOT_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz"
export WANDB_DIR="$ROOT_DIR/artifacts/wandb"
mkdir -p "$WANDB_DIR"

echo "Initializing WandB sweep..."
SWEEP_OUTPUT=$(wandb sweep config/sweeps/vae_wandb_sweep.yaml 2>&1)
echo "$SWEEP_OUTPUT"

# Extract sweep ID from output
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \K[^\s]+' | tail -1)

if [ -z "$SWEEP_ID" ]; then
    echo "Failed to extract sweep ID. Please check the output above."
    exit 1
fi

echo ""
echo "============================================="
echo "Sweep initialized successfully!"
echo "Sweep ID: $SWEEP_ID"
echo "============================================="
echo ""
echo "To run the sweep on SLURM, use:"
echo "sbatch --export=ALL,SWEEP_ID=$SWEEP_ID scripts/run_vae_sweep.sh"
echo ""
echo "Or for multiple parallel agents:"
echo "sbatch --export=ALL,SWEEP_ID=$SWEEP_ID --array=1-5 scripts/run_vae_sweep.sh"
