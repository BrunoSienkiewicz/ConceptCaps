#!/bin/bash -l
# Script to create MTG Jamendo tags dataset using Hydra
# 
# Usage:
#   ./scripts/run_create_mtg_jamendo_dataset.sh [HYDRA_OVERRIDES]
#   ./scripts/run_create_mtg_jamendo_dataset.sh mtg_jamendo.train_split=0.9
#   ./scripts/run_create_mtg_jamendo_dataset.sh --config-name create_mtg_jamendo_dataset

set -e

# Parse arguments
HYDRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      echo "Usage: ./scripts/run_create_mtg_jamendo_dataset.sh [HYDRA_OVERRIDES]"
      echo ""
      echo "Examples:"
      echo "  # Create dataset with default settings"
      echo "  ./scripts/run_create_mtg_jamendo_dataset.sh"
      echo ""
      echo "  # Custom train/val split"
      echo "  ./scripts/run_create_mtg_jamendo_dataset.sh mtg_jamendo.train_split=0.9 mtg_jamendo.val_split=0.05"
      echo ""
      echo "  # Custom output directory"
      echo "  ./scripts/run_create_mtg_jamendo_dataset.sh mtg_jamendo.output_dir=data/custom_jamendo"
      echo ""
      exit 0
      ;;
    *)
      HYDRA_OVERRIDES+=("$1")
      shift
      ;;
  esac
done

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

echo "Project root: $PROJECT_ROOT"
echo "Creating MTG Jamendo dataset..."
echo ""

# Export environment variables
export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"
export PROJECT_ROOT="${PROJECT_ROOT}"

# Run the script
python -m src.caption.create_mtg_jamendo \
    --config-name create_mtg_jamendo_dataset \
    "${HYDRA_OVERRIDES[@]}"

echo ""
echo "✓ Dataset creation complete!"
