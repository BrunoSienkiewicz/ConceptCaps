#!/bin/bash -l
#SBATCH -J jupyter_notebook
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="logs/jupyter_%j.out"
#SBATCH --error="logs/jupyter_%j.err"

# Activate your conda environment
conda activate music-gen-interpretability3

# Get the compute node hostname
NODE=$(hostname)
PORT=8888

echo "Jupyter will run on node: $NODE"
echo "Port: $PORT"
echo ""
echo "On your LOCAL machine, run:"
echo "ssh -N -L $PORT:$NODE:$PORT <your_username>@<login_node>"
echo ""

# Start Jupyter without opening browser
jupyter notebook --no-browser --port=$PORT --ip=0.0.0.0
