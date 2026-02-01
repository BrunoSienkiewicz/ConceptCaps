#!/bin/bash -l
#SBATCH -J make_env
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH -A ${SLURM_ACCOUNT:-your-slurm-account}
#SBATCH -p ${SLURM_PARTITION:-plgrid-gpu-a100}
#SBATCH --output="output.out"
#SBATCH --error="error.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

CONDA_DIR="${PLG_GROUPS_STORAGE:-/path/to/storage}/${PLG_GROUP:-your-group}/$USER/.conda"

module load Miniconda3/23.3.1
conda init --all

# Configure conda to use a custom directory for packages and environments
conda config --add pkgs\_dirs $CONDA_DIR/pkgs
conda config --add envs\_dirs $CONDA_DIR/envs

# Use libmamba solver for faster environment resolution
conda config --set solver libmamba

conda install -n musicgen-interpretability pytorch-gpu torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia
