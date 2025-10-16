#!/bin/bash -l
#SBATCH -J run_tta
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

OUT_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz"
CONDA_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz/.conda"
ENV_DIR="$CONDA_DIR/envs/$(cat environment.yml | grep -E "name: " | cut -d " " -f 2)"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME="$OUT_DIR/.cache/huggingface"

mkdir -p "$OUT_DIR/.cache/huggingface"
srun "$ENV_DIR/bin/python" src/scripts/run_tcav.py +preset="$PRESET"
