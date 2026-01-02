#!/bin/bash
# Comparison of different parallelization strategies for TTA generation

cat << 'EOF'
================================================================================
TTA Generation Parallelization Strategy Comparison
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│ Strategy 1: SEQUENTIAL (Original)                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ File:           scripts/run_all_tta.sh                                      │
│ GPUs:           1                                                            │
│ Batch Size:     32                                                           │
│ Time:           ~10 hours (all datasets)                                     │
│ Total GPU-hours: 10                                                          │
│                                                                              │
│ Run:            sbatch scripts/run_all_tta.sh                               │
│                                                                              │
│ ✓ Simple setup                                                               │
│ ✓ Minimal resources                                                          │
│ ✗ Slowest option                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Strategy 2: MULTI-GPU (New - Best for Single Node)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│ File:           scripts/run_all_tta_2gpu.sh                                 │
│ GPUs:           2 (both used per dataset)                                    │
│ Batch Size:     64 (32 per GPU)                                              │
│ Time:           ~5 hours (all datasets)                                      │
│ Total GPU-hours: 10                                                          │
│                                                                              │
│ Run:            sbatch scripts/run_all_tta_2gpu.sh                          │
│                                                                              │
│ ✓ 2x faster than sequential                                                 │
│ ✓ Better GPU utilization                                                     │
│ ✓ Larger effective batch size                                               │
│ ✓ Same total GPU-hours                                                      │
│ ○ Requires 2 GPUs on same node                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Strategy 3: DATASET PARALLELIZATION (New - Best if Many GPUs Available)     │
├─────────────────────────────────────────────────────────────────────────────┤
│ File:           scripts/run_all_tta_parallel.sh (Job Array)                 │
│                 scripts/submit_tta_jobs.sh (Separate Jobs)                  │
│ GPUs:           5 (one per dataset)                                          │
│ Batch Size:     32 per dataset                                               │
│ Time:           ~2 hours (longest dataset)                                   │
│ Total GPU-hours: 10                                                          │
│                                                                              │
│ Run (Array):    sbatch scripts/run_all_tta_parallel.sh                      │
│ Run (Separate): bash scripts/submit_tta_jobs.sh                             │
│                                                                              │
│ ✓ 5x faster than sequential (if GPUs available)                             │
│ ✓ Independent jobs - easy to rerun                                          │
│ ✓ Best utilization if cluster has many GPUs                                 │
│ ○ Requires 5 GPUs simultaneously                                            │
│ ○ May need to wait in queue                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Strategy 4: HYBRID (Best of Both Worlds)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Combine multi-GPU + dataset parallelization                                 │
│ GPUs:           10 total (2 per dataset, 5 datasets)                         │
│ Batch Size:     64 per dataset (32 per GPU)                                  │
│ Time:           ~1 hour (longest dataset)                                    │
│ Total GPU-hours: 10                                                          │
│                                                                              │
│ Manual setup required - see instructions below                              │
│                                                                              │
│ ✓ 10x faster than sequential                                                │
│ ✓ Maximum throughput                                                         │
│ ✗ Requires 10 GPUs                                                           │
│ ✗ Most complex setup                                                         │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
Recommendations
================================================================================

📊 Have 2 GPUs available?
   → Use Strategy 2 (Multi-GPU)
   → Command: sbatch scripts/run_all_tta_2gpu.sh
   → Time: ~5 hours

🚀 Have 5+ GPUs available?
   → Use Strategy 3 (Dataset Parallelization)
   → Command: sbatch scripts/run_all_tta_parallel.sh
   → Time: ~2 hours

💪 Have 10 GPUs available?
   → Use Strategy 4 (Hybrid) - see setup below
   → Time: ~1 hour

💰 Limited resources?
   → Use Strategy 1 (Sequential)
   → Command: sbatch scripts/run_all_tta.sh
   → Time: ~10 hours

================================================================================
Hybrid Setup (Strategy 4)
================================================================================

Modify run_all_tta_parallel.sh to use 2 GPUs per task:

1. Edit the array script:
   #SBATCH --gres=gpu:2  # Change from gpu:1 to gpu:2

2. Change the python script call:
   srun python src/scripts/tta/run_tta_generation_parallel.py \\
       +preset=tta/plgrid_musiccaps \\
       data=\$DATASET \\
       data.batch_size=64

3. Submit:
   sbatch scripts/run_all_tta_parallel.sh

This runs 5 array tasks, each with 2 GPUs, processing all datasets in parallel.

================================================================================
Quick Decision Guide
================================================================================

How many GPUs can you allocate RIGHT NOW on your cluster?

  1 GPU  → scripts/run_all_tta.sh (10h)
  2 GPUs → scripts/run_all_tta_2gpu.sh (5h)  ⭐ RECOMMENDED
  5 GPUs → scripts/run_all_tta_parallel.sh (2h)
  10 GPUs → Hybrid approach (1h)

The 2-GPU approach (Strategy 2) offers the best balance of:
- Speed improvement (2x)
- Resource efficiency
- Setup simplicity
- Availability on most clusters

================================================================================

EOF
