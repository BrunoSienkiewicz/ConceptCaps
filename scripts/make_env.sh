#!/bin/bash -l
#SBATCH -J make_env
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="output.out"
#SBATCH --error="error.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

CONDA_DIR="$PLG_GROUPS_STORAGE/plgbsienkiewicz/.conda"

module load Miniconda3/23.3.1
conda init --all
conda config --add pkgs\_dirs $CONDA_DIR/pkgs
conda config --add envs\_dirs $CONDA_DIR/envs

srun make env
