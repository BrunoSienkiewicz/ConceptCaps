#!/bin/bash
# Initialize WandB sweep and get sweep ID
# Usage: ./scripts/init_wandb_sweep.sh [vae|caption]

cd "$(dirname "$0")/.." || exit 1

# Parse sweep type argument (default to vae)
SWEEP_TYPE="${1:-vae}"

if [ "$SWEEP_TYPE" != "vae" ] && [ "$SWEEP_TYPE" != "caption" ] && [ "$SWEEP_TYPE" != "tta" ]; then
    echo "Error: Invalid sweep type. Must be 'vae', 'caption', or 'tta'"
    echo "Usage: ./scripts/init_wandb_sweep.sh [vae|caption|tta]"
    exit 1
fi

# Set sweep config and script based on type
if [ "$SWEEP_TYPE" = "vae" ]; then
    SWEEP_CONFIG="config/sweeps/vae_wandb_sweep.yaml"
    SWEEP_SCRIPT="scripts/run_vae_sweep.sh"
    SWEEP_NAME="VAE Training"
fi
if [ "$SWEEP_TYPE" = "caption" ]; then
    SWEEP_CONFIG="config/sweeps/caption_wandb_sweep.yaml"
    SWEEP_SCRIPT="scripts/run_caption_sweep.sh"
    SWEEP_NAME="Caption Fine-Tuning"
fi
if [ "$SWEEP_TYPE" = "tta" ]; then
    SWEEP_CONFIG="config/sweeps/tta_wandb_sweep.yaml"
    SWEEP_SCRIPT="scripts/run_tta_sweep.sh"
    SWEEP_NAME="TTA Generation"
fi

# Activate conda environment
CONDA_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz/.conda"
ENV_NAME=$(grep -E '^name:' environment.yml | awk '{print $2}')
conda activate "$ENV_NAME"

# Set WandB directory
ROOT_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz"
export WANDB_DIR="$ROOT_DIR/artifacts/wandb"
mkdir -p "$WANDB_DIR"

echo "Initializing WandB sweep for $SWEEP_NAME..."
echo "Using config: $SWEEP_CONFIG"
SWEEP_OUTPUT=$(wandb sweep "$SWEEP_CONFIG" 2>&1)
echo "$SWEEP_OUTPUT"

# Extract sweep ID from output
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \K[^\s]+' | tail -1)

if [ -z "$SWEEP_ID" ]; then
    echo "Failed to extract sweep ID. Please check the output above."
    exit 1
fi

echo ""
echo "============================================="
echo "$SWEEP_NAME Sweep initialized successfully!"
echo "Sweep ID: $SWEEP_ID"
echo "============================================="
echo ""
echo "To run the sweep on SLURM, use:"
echo "sbatch --export=ALL,SWEEP_ID=$SWEEP_ID $SWEEP_SCRIPT"
echo ""
echo "Or for multiple parallel agents:"
echo "sbatch --export=ALL,SWEEP_ID=$SWEEP_ID --array=1-5 $SWEEP_SCRIPT"
