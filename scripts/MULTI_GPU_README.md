# Multi-GPU TTA Generation

This guide shows how to use **2 GPUs** to speed up audio generation by increasing batch size and parallelizing model inference.

## Quick Start

### Run all datasets with 2 GPUs:
```bash
sbatch scripts/run_all_tta_2gpu.sh
```

### Run single dataset with 2 GPUs:
```bash
sbatch scripts/run_tta_2gpu.sh -p tta/plgrid_musiccaps data=tta/musiccaps_captions
```

## Performance Improvements

| Configuration | GPUs | Batch Size | Speed |
|--------------|------|------------|-------|
| Original | 1 | 32 | ~10 hours |
| **Multi-GPU** | **2** | **64** | **~5 hours** (2x faster) |

### Why 2x Speedup?
- **Double batch size**: 64 total (32 per GPU) vs 32
- **Parallel computation**: Both GPUs generate simultaneously
- **Better memory utilization**: 2x GPU memory available

## How It Works

The multi-GPU implementation uses **PyTorch DataParallel**:

1. **Model Replication**: Model is copied to both GPUs
2. **Batch Splitting**: Input batch is split across GPUs automatically
3. **Parallel Generation**: Each GPU generates audio for its mini-batch
4. **Output Gathering**: Results are combined on the primary GPU

### Architecture
```
Input Batch (64) → Split
                   ├─ GPU 0: Process 32 samples → Generate audio
                   └─ GPU 1: Process 32 samples → Generate audio
                                                   ↓
                                            Combine outputs
```

## Scripts Overview

### 1. `run_all_tta_2gpu.sh` - Run All Datasets
Runs all 5 datasets sequentially using 2 GPUs for each.

**Resource allocation:**
- 2 GPUs (A100)
- 16GB RAM per CPU
- 8 CPUs (more for data loading)
- Batch size: 64 (32 per GPU)

**Usage:**
```bash
sbatch scripts/run_all_tta_2gpu.sh
```

**Datasets processed:**
1. Zero-shot captions
2. Base VAE captions
3. Fine-tuned VAE captions
4. MusicCaps captions
5. LP-MusicCaps captions

### 2. `run_tta_2gpu.sh` - Run Single Dataset
Flexible script for running individual datasets with custom settings.

**Usage:**
```bash
# Basic usage
sbatch scripts/run_tta_2gpu.sh -p tta/plgrid_musiccaps data=tta/musiccaps_captions

# Custom batch size
sbatch scripts/run_tta_2gpu.sh -p tta/plgrid_musiccaps -b 96 data=tta/musiccaps_captions

# Pass additional Hydra overrides
sbatch scripts/run_tta_2gpu.sh -p tta/plgrid_musiccaps model.model.max_new_tokens=512 data=tta/base_vae_captions
```

**Options:**
- `-p, --preset`: Hydra preset to use
- `-b, --batch-size`: Total batch size (default: 64)

## Implementation Details

### Files Created

1. **`src/tta/audio_parallel.py`** - Multi-GPU audio generation utilities
   - `generate_audio_samples_parallel()` - Main DataParallel implementation
   - `generate_audio_samples_manual_split()` - Alternative manual splitting approach
   - `DataParallelGeneration` - Wrapper class for proper generation handling

2. **`src/scripts/tta/run_tta_generation_parallel.py`** - Multi-GPU generation script
   - Detects available GPUs
   - Configures DataParallel
   - Handles batch size distribution

### Key Features

✅ **Automatic GPU detection**: Uses all allocated GPUs  
✅ **Batch size scaling**: Automatically distributes batches  
✅ **Graceful degradation**: Works with 1 GPU if 2 aren't available  
✅ **Memory efficient**: Proper cleanup and memory management  

## Batch Size Tuning

### Recommended Batch Sizes (Total)

| GPUs | Memory per GPU | Recommended Batch Size |
|------|----------------|------------------------|
| 1 | 40GB (A100) | 32-48 |
| 2 | 40GB (A100) | 64-96 |

### Finding Optimal Batch Size

1. Start with 64 (32 per GPU)
2. Monitor GPU memory usage
3. Increase gradually if memory allows:
   ```bash
   sbatch scripts/run_tta_2gpu.sh -b 96 data=tta/musiccaps_captions
   ```
4. Watch for OOM (Out of Memory) errors

### Memory Considerations

- **MusicGen Small**: ~6GB per model copy
- **Peak memory**: During generation (audio tensors)
- **Buffer**: Leave 10-15% free for safety

## Monitoring

### Check GPU usage:
```bash
# On compute node (during job)
nvidia-smi

# Watch in real-time
watch -n 1 nvidia-smi
```

### Check job status:
```bash
squeue -u $USER
```

### View logs:
```bash
# Live monitoring
tail -f logs/run_tta_2gpu_<job_id>.out

# Check errors
tail -f logs/run_tta_2gpu_<job_id>.err
```

## Troubleshooting

### OOM (Out of Memory) Errors

**Solution 1: Reduce batch size**
```bash
sbatch scripts/run_tta_2gpu.sh -b 48 data=tta/musiccaps_captions
```

**Solution 2: Use gradient checkpointing**
Add to model config:
```yaml
model:
  model:
    use_cache: false
```

### Only 1 GPU Available

The script gracefully handles this - it will use single GPU with smaller effective batch size.

### DataParallel Issues

If DataParallel causes problems, the code includes an alternative manual splitting approach in `audio_parallel.py`. Switch to `generate_audio_samples_manual_split()` if needed.

### Uneven Batch Sizes

If batch size isn't evenly divisible by number of GPUs, the last GPU gets the remainder. This is handled automatically.

## Combining Strategies

You can combine multi-GPU with dataset parallelization from the previous solution:

1. **Option A**: 2 GPUs per job, 2-3 jobs in parallel
   - 4-6 GPUs total
   - Fastest approach if GPUs available

2. **Option B**: 2 GPUs per job, sequential jobs
   - More memory efficient
   - Still 2x faster than original

## Alternative: Accelerate Library

For more advanced multi-GPU setups, consider using HuggingFace Accelerate:

```python
from accelerate import Accelerator
accelerator = Accelerator()
model = accelerator.prepare(model)
```

This provides better support for:
- Mixed precision training
- Distributed training (multi-node)
- Gradient accumulation
- DeepSpeed integration

## Performance Tips

1. **Increase dataloader workers**: More CPUs = faster data loading
   ```yaml
   data:
     dataloader_num_workers: 8
   ```

2. **Use mixed precision**: Faster computation, less memory
   ```yaml
   model:
     dtype: float16
   ```

3. **Pin memory**: Faster GPU transfers
   ```yaml
   data:
     pin_memory: true
   ```

4. **Prefetch**: Overlap data loading with computation
   ```yaml
   data:
     prefetch_factor: 2
   ```
