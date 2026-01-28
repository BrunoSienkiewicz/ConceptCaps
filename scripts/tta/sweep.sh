#!/bin/bash -l
#SBATCH -J tta_sweep
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="logs/tta_sweep_%j.out"
#SBATCH --error="logs/tta_sweep_%j.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

ROOT_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz"
OUT_DIR="$ROOT_DIR/tta_sweep_results"
PLGRID_ARTIFACTS_DIR="$ROOT_DIR/artifacts"
CONDA_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz/.conda"
ENV_DIR="$CONDA_DIR/envs/$(grep -E '^name:' environment.yml | awk '{print $2}')"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export OUT_DIR="$OUT_DIR"
export PLGRID_ARTIFACTS_DIR="$PLGRID_ARTIFACTS_DIR"
export CONDA_DIR="$CONDA_DIR"
export WANDB_DIR="$PLGRID_ARTIFACTS_DIR/wandb"

SWEEP_CONFIG="config/sweeps/tta_wandb_sweep.yaml"

conda activate music-gen-interpretability3

echo "Initializing WandB sweep"
echo "Using config: $SWEEP_CONFIG"
SWEEP_OUTPUT=$(wandb sweep "$SWEEP_CONFIG" 2>&1)
echo "$SWEEP_OUTPUT"

# Extract sweep ID from output
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \K[^\s]+' | tail -1)

echo "Starting WandB agent for sweep: $SWEEP_ID"
srun wandb agent "$SWEEP_ID"
