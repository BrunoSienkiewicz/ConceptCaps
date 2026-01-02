#!/bin/bash -l
#SBATCH -J run_all_tta_2gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --gres=gpu:2  # Request 2 GPUs
#SBATCH --cpus-per-task=8  # More CPUs for data loading
#SBATCH --time=05:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="logs/run_all_tta_2gpu_%j.out"
#SBATCH --error="logs/run_all_tta_2gpu_%j.err"

cd "$SLURM_SUBMIT_DIR" || exit 1

ROOT_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz"
PLGRID_ARTIFACTS_DIR="$ROOT_DIR/artifacts"
CONDA_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz/.conda"
ENV_DIR="$CONDA_DIR/envs/$(grep -E '^name:' environment.yml | awk '{print $2}')"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export PLGRID_ARTIFACTS_DIR="$PLGRID_ARTIFACTS_DIR"
export CONDA_DIR="$CONDA_DIR"
export WANDB_DIR="$PLGRID_ARTIFACTS_DIR/wandb"

# Make both GPUs visible
export CUDA_VISIBLE_DEVICES=0,1

conda activate music-gen-interpretability3

echo "=========================================="
echo "Running TTA Generation - All Datasets (2 GPUs)"
echo "Available GPUs: $(echo $CUDA_VISIBLE_DEVICES)"
echo "Batch size increased to 64 (32 per GPU)"
echo "=========================================="

# 1. Zero-shot captions
echo ""
echo "1/5: Running TTA generation with zero-shot captions..."
srun $ENV_DIR/bin/python src/scripts/tta/run_tta_generation_parallel.py +preset=tta/plgrid_musiccaps \
    data=tta/zero_shot_captions \
    data.batch_size=64

# 2. Base VAE captions
echo ""
echo "2/5: Running TTA generation with base VAE captions..."
srun $ENV_DIR/bin/python src/scripts/tta/run_tta_generation_parallel.py +preset=tta/plgrid_musiccaps \
    data=tta/base_vae_captions \
    data.batch_size=64

# 3. Fine-tuned VAE captions
echo ""
echo "3/5: Running TTA generation with fine-tuned VAE captions..."
srun $ENV_DIR/bin/python src/scripts/tta/run_tta_generation_parallel.py +preset=tta/plgrid_musiccaps \
    data=tta/ft_vae_captions \
    data.batch_size=64

# 4. MusicCaps captions
echo ""
echo "4/5: Running TTA generation with MusicCaps captions..."
srun $ENV_DIR/bin/python src/scripts/tta/run_tta_generation_parallel.py +preset=tta/plgrid_musiccaps \
    data=tta/musiccaps_captions \
    data.batch_size=64

# 5. LP-MusicCaps captions
echo ""
echo "5/5: Running TTA generation with LP-MusicCaps captions..."
srun $ENV_DIR/bin/python src/scripts/tta/run_tta_generation_parallel.py +preset=tta/plgrid_musiccaps \
    data=tta/lp-musiccaps_captions \
    data.batch_size=64

echo ""
echo "=========================================="
echo "All TTA generation runs completed!"
echo "=========================================="
