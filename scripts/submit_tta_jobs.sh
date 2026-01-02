#!/bin/bash
# Alternative approach: Submit separate independent jobs for each dataset
# This gives more flexibility and easier monitoring

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Define datasets
DATASETS=(
    "tta/zero_shot_captions"
    "tta/base_vae_captions"
    "tta/ft_vae_captions"
    "tta/musiccaps_captions"
    "tta/lp-musiccaps_captions"
)

DATASET_NAMES=(
    "zero_shot"
    "base_vae"
    "ft_vae"
    "musiccaps"
    "lp_musiccaps"
)

echo "=========================================="
echo "Submitting parallel TTA generation jobs"
echo "=========================================="

JOB_IDS=()

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    NAME="${DATASET_NAMES[$i]}"
    
    echo ""
    echo "Submitting job for: $NAME"
    
    JOB_ID=$(sbatch \
        --job-name="tta_$NAME" \
        --output="logs/tta_${NAME}_%j.out" \
        --error="logs/tta_${NAME}_%j.err" \
        scripts/run_tta.sh -p tta/plgrid_musiccaps data=$DATASET | awk '{print $4}')
    
    if [ -n "$JOB_ID" ]; then
        JOB_IDS+=("$JOB_ID")
        echo "Job submitted with ID: $JOB_ID"
    else
        echo "Failed to submit job for $NAME"
    fi
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "Job IDs: ${JOB_IDS[*]}"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check specific job: squeue -j <job_id>"
echo "Cancel all jobs: scancel ${JOB_IDS[*]}"
