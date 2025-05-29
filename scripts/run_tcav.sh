#!/bin/bash -l
#SBATCH -J run_tcav
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="output.out"
#SBATCH --error="error.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

CONDA_DIR="/net/tscratch/people/plgbsienkiewicz/.conda"
ENV_DIR="$CONDA_DIR/envs/$(cat environment.yml | grep -E "name: " | cut -d " " -f 2)"

srun /bin/hostname
module load Miniconda3/23.3.1
conda init --all
conda config --add pkgs\_dirs $CONDA_DIR/pkgs
conda config --add envs\_dirs $CONDA_DIR/envs

if [ ! -d "$ENV_DIR" ]; then
  make env
fi

conda activate "$ENV_DIR"
export PYTHONPATH="$PYTHONPATH:$(pwd)"
"$ENV_DIR/bin/python" src/scripts/run_tcav.py +preset=plgird +logger=wandb
