#!/bin/bash -l
#SBATCH -J run_tta_sweep
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="logs/run_tta_sweep_%j.out"
#SBATCH --error="logs/run_tta_sweep_%j.err"
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

conda activate music-gen-interpretability3

# Check if SWEEP_ID is provided as an environment variable
if [ -z "$SWEEP_ID" ]; then
    echo "Error: SWEEP_ID environment variable not set."
    echo "First, initialize the sweep with: wandb sweep config/sweeps/tta_wandb_sweep.yaml"
    echo "Then run this script with: sbatch --export=ALL,SWEEP_ID=<your-sweep-id> scripts/run_tta_sweep.sh"
    exit 1
fi

echo "Starting WandB agent for sweep: $SWEEP_ID"
srun wandb agent "$SWEEP_ID"
