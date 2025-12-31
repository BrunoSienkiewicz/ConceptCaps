#!/bin/bash -l
#SBATCH -J run_all_tta
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="logs/run_all_tta_%j.out"
#SBATCH --error="logs/run_all_tta_%j.err"

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

conda activate music-gen-interpretability3

echo "=========================================="
echo "Running TTA Generation - All Datasets"
echo "=========================================="

# 1. Zero-shot captions
# echo ""
# echo "1/5: Running TTA generation with zero-shot captions..."
# srun $ENV_DIR/bin/python src/scripts/tta/run_tta_generation.py +preset=tta/plgrid_musiccaps \
#     data=tta/zero_shot_captions

# # 2. Base VAE captions
# echo ""
# echo "2/5: Running TTA generation with base VAE captions..."
# srun $ENV_DIR/bin/python src/scripts/tta/run_tta_generation.py +preset=tta/plgrid_musiccaps \
#     data=tta/base_vae_captions

# # 3. Fine-tuned VAE captions
# echo ""
# echo "3/5: Running TTA generation with fine-tuned VAE captions..."
# srun $ENV_DIR/bin/python src/scripts/tta/run_tta_generation.py +preset=tta/plgrid_musiccaps \
#     data=tta/ft_vae_captions

# 4. MusicCaps captions
echo ""
echo "4/5: Running TTA generation with LP-MusicCaps captions..."
srun $ENV_DIR/bin/python src/scripts/tta/run_tta_generation.py +preset=tta/plgrid_musiccaps \
    data=tta/musiccaps_captions

# 5. LP-MusicCaps captions
echo ""
echo "5/5: Running TTA generation with LP-MusicCaps captions..."
srun $ENV_DIR/bin/python src/scripts/tta/run_tta_generation.py +preset=tta/plgrid_musiccaps \
    data=tta/lp-musiccaps_captions

echo ""
echo "=========================================="
echo "All TTA generation runs completed!"
echo "=========================================="
