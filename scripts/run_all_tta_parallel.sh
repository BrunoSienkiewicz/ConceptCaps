#!/bin/bash -l
#SBATCH -J run_all_tta_parallel
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --array=0-4  # 5 datasets (0-4)
#SBATCH --output="logs/run_all_tta_parallel_%A_%a.out"
#SBATCH --error="logs/run_all_tta_parallel_%A_%a.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

ROOT_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz"
PLGRID_ARTIFACTS_DIR="$ROOT_DIR/artifacts"
CONDA_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz/.conda"
ENV_DIR="$CONDA_DIR/envs/$(grep -E '^name:' environment.yml | awk '{print $2}')"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export PLGRID_ARTIFACTS_DIR="$PLGRID_ARTIFACTS_DIR"
export CONDA_DIR="$CONDA_DIR"
export WANDB_DIR="$PLGRID_ARTIFACTS_DIR/wandb"

conda activate music-gen-interpretability3

# Define datasets array
DATASETS=(
    "tta/zero_shot_captions"
    "tta/base_vae_captions"
    "tta/ft_vae_captions"
    "tta/musiccaps_captions"
    "tta/lp-musiccaps_captions"
)

# Get the dataset for this array task
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo "=========================================="

# Run TTA generation for this dataset
srun $ENV_DIR/bin/python src/scripts/tta/run_tta_generation.py +preset=tta/plgrid_musiccaps \
    data=$DATASET

echo ""
echo "=========================================="
echo "TTA generation completed for $DATASET"
echo "=========================================="
