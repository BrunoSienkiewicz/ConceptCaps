#!/bin/bash -l
#SBATCH -J run_caption_fine_tuning
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output=logs/caption_fine_tuning.out
#SBATCH --error=logs/caption_fine_tuning.err

cd "$SLURM_SUBMIT_DIR" || exit 1

PRESET=""
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

mkdir -p "$HF_HOME"
srun "$ENV_DIR/bin/python" src/scripts/caption/run_fine_tuning_lightning.py "${HYDRA_OVERRIDES[@]}"