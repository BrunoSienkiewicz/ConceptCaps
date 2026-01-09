#!/bin/bash -l
#SBATCH -J run_all_inference
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="logs/run_all_inference_%j.out"
#SBATCH --error="logs/run_all_inference_%j.err"

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
echo "Running Caption Inference - All Configs"
echo "=========================================="

# # 1. Zero-shot with original dataset
# echo ""
# echo "1/5: Running zero-shot inference on original dataset..."
# srun $ENV_DIR/bin/python src/scripts/caption/run_inference_lightning.py \
#     +preset=caption/zero_shot_plgrid

# # 2. Zero-shot with random tags dataset
# echo ""
# echo "2/5: Running zero-shot inference on random tags dataset..."
# srun $ENV_DIR/bin/python src/scripts/caption/run_inference_lightning.py \
#     +preset=caption/llama_inference_plgrid \
#     data=caption/load_random-tags-dataset

# # 3. Zero-shot with VAE tags dataset
# echo ""
# echo "3/5: Running zero-shot inference on VAE tags dataset..."
# srun $ENV_DIR/bin/python src/scripts/caption/run_inference_lightning.py \
#     +preset=caption/llama_inference_plgrid \
#     data=caption/load_vae-tags-dataset

# # 4. Fine-tuned with random tags dataset
# echo ""
# echo "4/5: Running fine-tuned inference on random tags dataset..."
# srun $ENV_DIR/bin/python src/scripts/caption/run_inference_lightning.py \
#     +preset=caption/llama_inference_plgrid \
#     model.checkpoint_dir="/net/pr2/projects/plgrid/plggailpwln/plgbsienkiewicz/artifacts/models/meta-llama/Llama-3.1-8B-Instruct/2025-12-31_22-36-38/lora_adapter" \
#     data=caption/load_random-tags-dataset

# 5. Fine-tuned with VAE tags dataset
echo ""
echo "5/5: Running fine-tuned inference on VAE tags dataset..."
srun $ENV_DIR/bin/python src/scripts/caption/run_inference_lightning.py \
    +preset=caption/llama_inference_plgrid \
    model.checkpoint_dir="/net/pr2/projects/plgrid/plggailpwln/plgbsienkiewicz/artifacts/models/meta-llama/Llama-3.1-8B-Instruct/2026-01-08_20-57-32/lora_adapter" \
    data=caption/load_vae-tags-dataset

echo ""
echo "=========================================="
echo "All inference runs completed!"
echo "=========================================="