# Parallel TTA Generation

This directory contains scripts for running TTA generation in parallel to reduce processing time from ~10 hours to ~2 hours (for the longest dataset).

## Approach 1: SLURM Job Array (Recommended)

**Best for:** Automatic load balancing, single submission, cleaner job management

```bash
sbatch scripts/run_all_tta_parallel.sh
```

This will:
- Submit a job array with 5 tasks (one per dataset)
- Each task runs on its own GPU independently
- All tasks start simultaneously (resources permitting)
- Total time: ~2 hours (time of longest dataset)

### Monitor progress:
```bash
# Check all array tasks
squeue -u $USER

# Check specific job array
squeue -j <job_id>

# Cancel entire job array
scancel <job_id>
```

### Output files:
- `logs/run_all_tta_parallel_<job_id>_<task_id>.out`
- `logs/run_all_tta_parallel_<job_id>_<task_id>.err`

Where `<task_id>` is 0-4 for each dataset.

## Approach 2: Separate Jobs

**Best for:** More control, easier to rerun individual datasets, different resource requirements

```bash
bash scripts/submit_tta_jobs.sh
```

This will:
- Submit 5 separate independent jobs
- Each job has its own ID for easy monitoring
- Can customize resources per dataset if needed

### Monitor progress:
```bash
# Check all jobs
squeue -u $USER

# Cancel specific job
scancel <job_id>

# Cancel all (job IDs printed after submission)
scancel <job_id_1> <job_id_2> ...
```

### Output files:
- `logs/tta_zero_shot_<job_id>.out`
- `logs/tta_base_vae_<job_id>.out`
- `logs/tta_ft_vae_<job_id>.out`
- `logs/tta_musiccaps_<job_id>.out`
- `logs/tta_lp_musiccaps_<job_id>.out`

## Datasets Processed

Both approaches process these 5 datasets in parallel:

1. `tta/zero_shot_captions` - Zero-shot generated captions
2. `tta/base_vae_captions` - Base VAE model captions
3. `tta/ft_vae_captions` - Fine-tuned VAE model captions
4. `tta/musiccaps_captions` - Original MusicCaps captions
5. `tta/lp-musiccaps_captions` - LP-MusicCaps captions

## Resource Requirements

Each job requires:
- 1 GPU (A100)
- 16GB RAM
- 4 CPUs
- ~2 hours max (adjust `--time` if needed)

Total resources needed: **5 GPUs** running simultaneously.

## Comparison

| Approach | Pros | Cons |
|----------|------|------|
| **Job Array** | Single submission, cleaner logs, automatic management | Less flexible for per-dataset customization |
| **Separate Jobs** | Independent control, easier resubmission, custom resources | More manual management, 5 separate submissions |

## Troubleshooting

**Not enough GPUs available?**
- Adjust `--array=0-2` to run fewer tasks at once
- Or use separate jobs and submit in batches

**One dataset fails?**
- Job Array: Resubmit with `--array=<failed_task_id>`
- Separate Jobs: Just resubmit that specific dataset

**Need different resources per dataset?**
- Use separate jobs approach
- Modify resource requests in [run_tta.sh](scripts/run_tta.sh) per dataset
