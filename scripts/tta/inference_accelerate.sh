#!/bin/bash -l
#SBATCH -J tta_accelerate
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH -A ${SLURM_ACCOUNT:-your-slurm-account}
#SBATCH -p ${SLURM_PARTITION:-plgrid-gpu-a100}
#SBATCH --output="logs/tta_accelerate_%j.out"
#SBATCH --error="logs/tta_accelerate_%j.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

# Initialize variables with default values
PRESET="default"
HYDRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--preset)
      PRESET="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      HYDRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  HYDRA_OVERRIDES+=("$@")
fi

if [[ -n "$PRESET" ]]; then
  HYDRA_OVERRIDES+=("+preset=$PRESET")
fi

ROOT_DIR="${PLG_GROUPS_STORAGE:-/path/to/storage}/${PLG_GROUP:-your-group}/$USER"
OUT_DIR="$ROOT_DIR/caption_fine_tuning"
PLGRID_ARTIFACTS_DIR="$ROOT_DIR/artifacts"
CONDA_DIR="${PLG_GROUPS_STORAGE:-/path/to/storage}/${PLG_GROUP:-your-group}/$USER/.conda"
ENV_DIR="$CONDA_DIR/envs/$(grep -E '^name:' environment.yml | awk '{print $2}')"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export OUT_DIR="$OUT_DIR"
export PLGRID_ARTIFACTS_DIR="$PLGRID_ARTIFACTS_DIR"
export CONDA_DIR="$CONDA_DIR"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "$OUT_DIR/.cache/huggingface"

conda activate "$(grep -E '^name:' environment.yml | awk '{print $2}')"

srun accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    src/scripts/tta/inference.py "${HYDRA_OVERRIDES[@]}"
