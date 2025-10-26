#!/bin/bash -l
#SBATCH -J run_caption_generation
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="output.out"
#SBATCH --error="error.err"

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

OUT_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz"
CONDA_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz/.conda"
ENV_DIR="$CONDA_DIR/envs/$(grep -E '^name:' environment.yml | awk '{print $2}')"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME="$OUT_DIR/.cache/huggingface"
mkdir -p "$HF_HOME"

srun "$ENV_DIR/bin/python" src/scripts/run_caption_generation.py "${HYDRA_OVERRIDES[@]}"