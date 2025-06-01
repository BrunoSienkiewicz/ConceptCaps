#!/bin/bash -l
#SBATCH -J make_env
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="output.out"
#SBATCH --error="error.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

CONDA_DIR="/net/tscratch/people/plgbsienkiewicz/.conda"

module load Miniconda3/23.3.1
conda init --all
conda config --add pkgs\_dirs $CONDA_DIR/pkgs
conda config --add envs\_dirs $CONDA_DIR/envs

srun make env
