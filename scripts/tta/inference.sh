#!/bin/bash -l
#SBATCH -J tta
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="logs/tta_%j.out"
#SBATCH --error="logs/tta_%j.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

# Initialize variables with default values
PRESET="default"
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
  -p | --preset)
    PRESET="$2"
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

mkdir -p "$OUT_DIR/.cache/huggingface"
"$ENV_DIR/bin/python" src/scripts/tta/run_tta_generation.py +preset="$PRESET"
