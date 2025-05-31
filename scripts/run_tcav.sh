#!/bin/bash -l
#SBATCH -J run_tcav
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

TEMP_DIR="/net/tscratch/people/plgbsienkiewicz"
CONDA_DIR="$TEMP_DIR/.conda"
ENV_DIR="$CONDA_DIR/envs/$(cat environment.yml | grep -E "name: " | cut -d " " -f 2)"

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME="$TEMP_DIR/.cache/huggingface"
srun "$ENV_DIR/bin/python" src/scripts/run_tcav.py +preset=plgrid
