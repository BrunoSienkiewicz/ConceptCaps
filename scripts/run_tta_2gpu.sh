#!/bin/bash -l
#SBATCH -J run_tta_2gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --gres=gpu:2  # Request 2 GPUs
#SBATCH --cpus-per-task=8  # More CPUs for data loading
#SBATCH --time=03:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="logs/run_tta_2gpu_%j.out"
#SBATCH --error="logs/run_tta_2gpu_%j.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

# Initialize variables with default values
PRESET="default"
BATCH_SIZE=64  # Default batch size for 2 GPUs (32 per GPU)
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
  -p | --preset)
    PRESET="$2"
    shift # past argument
    shift # past value
    ;;
  -b | --batch-size)
    BATCH_SIZE="$2"
    shift # past argument
    shift # past value
    ;;
  -* | --*)
    echo "Unknown option $1"
    exit 1
    ;;
  *)
    POSITIONAL_ARGS+=("$1") # save positional arg
    shift                   # past argument
    ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

ROOT_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz"
OUT_DIR="$ROOT_DIR/caption_fine_tuning"
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

# Make both GPUs visible
export CUDA_VISIBLE_DEVICES=0,1

conda activate music-gen-interpretability3

echo "=========================================="
echo "Running TTA Generation (2 GPUs)"
echo "Available GPUs: $(echo $CUDA_VISIBLE_DEVICES)"
echo "Batch size: $BATCH_SIZE"
echo "=========================================="

srun "$ENV_DIR/bin/python" src/scripts/tta/run_tta_generation_parallel.py \
  +preset="$PRESET" \
  data.batch_size="$BATCH_SIZE" \
  "$@"
