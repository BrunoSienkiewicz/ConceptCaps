#!/bin/bash -l
#SBATCH -J caption_sweep
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH -A ${SLURM_ACCOUNT:-your-slurm-account}
#SBATCH -p ${SLURM_PARTITION:-plgrid-gpu-a100}
#SBATCH --output="logs/caption_sweep_%j.out"
#SBATCH --error="logs/caption_sweep_%j.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

ROOT_DIR="${PLG_GROUPS_STORAGE:-/path/to/storage}/${PLG_GROUP:-your-group}/$USER"
PLGRID_ARTIFACTS_DIR="$ROOT_DIR/artifacts"
CONDA_DIR="${PLG_GROUPS_STORAGE:-/path/to/storage}/${PLG_GROUP:-your-group}/$USER/.conda"
ENV_DIR="$CONDA_DIR/envs/$(grep -E '^name:' environment.yml | awk '{print $2}')"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export PLGRID_ARTIFACTS_DIR="$PLGRID_ARTIFACTS_DIR"
export CONDA_DIR="$CONDA_DIR"
export WANDB_DIR="$PLGRID_ARTIFACTS_DIR/wandb"

conda activate "$(grep -E '^name:' environment.yml | awk '{print $2}')"

SWEEP_CONFIG="config/sweeps/caption_wandb_sweep.yaml"

echo "Initializing WandB sweep"
echo "Using config: $SWEEP_CONFIG"
SWEEP_OUTPUT=$(wandb sweep "$SWEEP_CONFIG" 2>&1)
echo "$SWEEP_OUTPUT"

# Extract sweep ID from output
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP 'wandb agent \K[^\s]+' | tail -1)

echo "Starting WandB agent for caption fine-tuning sweep: $SWEEP_ID"
srun wandb agent "$SWEEP_ID"
