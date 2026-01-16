#!/bin/bash -l
#SBATCH -J full_pipeline
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH -A plgxailnpw25-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --output="logs/full_pipeline_%j.out"
#SBATCH --error="logs/full_pipeline_%j.err"

# =============================================================================
# Full Pipeline SLURM Script
# Runs: Fine-tuning → Inference → TTA Generation
# =============================================================================

cd "$SLURM_SUBMIT_DIR" || exit 1

# Initialize variables with default values
PRESET="plgrid"
STAGES=""
EXTRA_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
  -p | --preset)
    PRESET="$2"
    shift 2
    ;;
  -s | --stages)
    STAGES="$2"
    shift 2
    ;;
  --skip-finetuning)
    STAGES="inference,tta_generation"
    shift
    ;;
  --skip-inference)
    STAGES="fine_tuning,tta_generation"
    shift
    ;;
  --skip-tta)
    STAGES="fine_tuning,inference"
    shift
    ;;
  --finetuning-only)
    STAGES="fine_tuning"
    shift
    ;;
  --inference-only)
    STAGES="inference"
    shift
    ;;
  --tta-only)
    STAGES="tta_generation"
    shift
    ;;
  -* | --*)
    EXTRA_ARGS="$EXTRA_ARGS $1"
    shift
    ;;
  *)
    EXTRA_ARGS="$EXTRA_ARGS $1"
    shift
    ;;
  esac
done

# Setup paths
ROOT_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz"
OUT_DIR="$ROOT_DIR/full_pipeline"
PLGRID_ARTIFACTS_DIR="$ROOT_DIR/artifacts"
CONDA_DIR="$PLG_GROUPS_STORAGE/plggailpwln/plgbsienkiewicz/.conda"
ENV_NAME=$(grep -E '^name:' environment.yml | awk '{print $2}')
ENV_DIR="$CONDA_DIR/envs/$ENV_NAME"

# Export environment variables
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export HF_HOME="$ROOT_DIR/.cache/huggingface"
export OUT_DIR="$OUT_DIR"
export PLGRID_ARTIFACTS_DIR="$PLGRID_ARTIFACTS_DIR"
export CONDA_DIR="$CONDA_DIR"
export PROJECT_ROOT="$(pwd)"

# Create necessary directories
mkdir -p "$OUT_DIR"
mkdir -p "$ROOT_DIR/.cache/huggingface"
mkdir -p "logs"

# Print job information
echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"
echo "=============================================="
echo "Preset: $PRESET"
echo "Stages: ${STAGES:-all}"
echo "Extra args: $EXTRA_ARGS"
echo "=============================================="

# Build command
CMD="$ENV_DIR/bin/python src/scripts/run_full_pipeline.py"
CMD="$CMD +preset=pipeline/$PRESET"

# Add stages override if specified
if [[ -n "$STAGES" ]]; then
  # Convert comma-separated list to Hydra list format
  STAGES_LIST=$(echo "$STAGES" | sed 's/,/,/g')
  CMD="$CMD stages=[$STAGES_LIST]"
fi

# Add any extra arguments
if [[ -n "$EXTRA_ARGS" ]]; then
  CMD="$CMD $EXTRA_ARGS"
fi

echo "Executing: $CMD"
echo "=============================================="

# Run the pipeline
eval $CMD
EXIT_CODE=$?

echo "=============================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE
