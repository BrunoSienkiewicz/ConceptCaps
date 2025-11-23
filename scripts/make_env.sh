#!/bin/bash -l
#SBATCH -J make_env
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="output.out"
#SBATCH --error="error.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

CONDA_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz/.conda"

module load CUDA/12.4.0
module load GCCcore/12.2.0
module load CMake/3.24.3
module load GCC/11.2.0
module load Miniconda3/23.3.1-0
conda init --all

# Configure conda to use a custom directory for packages and environments
conda config --add pkgs\_dirs $CONDA_DIR/pkgs
conda config --add envs\_dirs $CONDA_DIR/envs

# Use libmamba solver for faster environment resolution
conda config --set solver libmamba

conda env create --file environment.yml
conda activate music-gen-interpretability2
pip install torch==2.6
