#!/bin/bash -l
#SBATCH -J run_vae_sweep
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="logs/run_vae_sweep.out"
#SBATCH --error="logs/run_vae_sweep.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

ROOT_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz"
OUT_DIR="$ROOT_DIR/vae_sweep_results"
PLGRID_ARTIFACTS_DIR="$ROOT_DIR/artifacts"
CONDA_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz/.conda"
ENV_DIR="$CONDA_DIR/envs/$(grep -E '^name:' environment.yml | awk '{print $2}')"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export OUT_DIR="$OUT_DIR"
export PLGRID_ARTIFACTS_DIR="$PLGRID_ARTIFACTS_DIR"
export CONDA_DIR="$CONDA_DIR"

conda activate music-gen-interpretability3
srun python -m hydra.main \
  --config-path=config/sweeps/vae_wandb_sweep.yaml \
  --config-name=vae_wandb_sweep \
  --multirun \
  src/scripts/vae/run_vae_training.py \
  wandb.dir="$PLGRID_ARTIFACTS_DIR/wandb" \
  paths=plgrid
