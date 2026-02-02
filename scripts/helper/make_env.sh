#!/bin/bash -l
#SBATCH -J make_env
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH -A ${SLURM_ACCOUNT:-your-slurm-account}
#SBATCH -p ${SLURM_PARTITION:-plgrid-gpu-a100}
#SBATCH --output="output.out"
#SBATCH --error="error.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

CONDA_DIR="${PLG_GROUPS_STORAGE:-/path/to/storage}/${PLG_GROUP:-your-group}/$USER/.conda"

module load CUDA/12.8.0
module load GCC/11.2.0
module load GCCcore/11.3.0
module load CMake/3.24.3
module load Miniconda3/23.3.1-0
conda init --all

# Configure conda to use a custom directory for packages and environments
conda config --add pkgs\_dirs $CONDA_DIR/pkgs
conda config --add envs\_dirs $CONDA_DIR/envs

# Use libmamba solver for faster environment resolution
conda config --set solver libmamba

conda env create --file environment.yml
conda activate "$(grep -E '^name:' environment.yml | awk '{print $2}')"
pip install torch==2.6


cd
if [ ! -d "bitsandbytes" ]; then
    git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
fi
cd bitsandbytes/
cmake -DCOMPUTE_BACKEND=cuda -DCOMPUTE_CAPABILITY=80 -S .
make
pip install -e .

cd
if [ ! -d "peft" ]; then
    git clone https://github.com/huggingface/peft
fi
cd peft
pip install -e .

cd
if [ ! -d "trl" ]; then
    git clone https://github.com/huggingface/trl.git
fi
cd trl/
pip install -e .
