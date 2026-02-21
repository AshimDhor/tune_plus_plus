# Reproducibility Guide

## Exact Experimental Setup

### Hardware
- **GPU**: NVIDIA A100 (40GB)
- **CPU**: AMD EPYC 7742 (64 cores)
- **RAM**: 512GB
- **Storage**: NVMe SSD

### Software Environment
```bash
Python 3.9.16
PyTorch 2.0.1+cu118
CUDA 11.8
cuDNN 8.7.0
MONAI 1.3.0
```

Full environment: See `requirements.txt`

### Random Seeds
```python
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Training Details

### Synapse Dataset
- **Training samples**: 18 CT scans
- **Validation samples**: 6 CT scans  
- **Test samples**: 6 CT scans
- **Epochs**: 1000
- **Batch size**: 2 (gradient accumulation: 4)
- **Effective batch size**: 8
- **Learning rate**: 1e-4
- **Optimizer**: AdamW (weight_decay=1e-4)
- **Scheduler**: Cosine annealing
- **Warmup epochs**: 50
- **Training time**: ~26 hours

### Data Augmentation
```python
RandCropByPosNegLabeld(
    spatial_size=(96, 96, 96),
    pos=1, neg=1,
    num_samples=2
)
RandFlipd(prob=0.5, spatial_axis=[0, 1, 2])
RandRotate90d(prob=0.5, spatial_axes=(0, 1))
RandScaleIntensityd(prob=0.5, factors=0.1)
RandShiftIntensityd(prob=0.5, offsets=0.1)
```

## Metrics Computation

### Dice Score
```python
def compute_dice(pred, target):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2 * intersection + 1e-5) / (union + 1e-5)
    return dice
```

### Betti Error
```python
def betti_error(pred, target):
    pred_betti = compute_betti_numbers(pred)
    target_betti = compute_betti_numbers(target)
    error = sum(abs(p - t) for p, t in zip(pred_betti, target_betti))
    return error
```

### ECE (Expected Calibration Error)
```python
def compute_ece(confidences, correctness, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            accuracy = correctness[in_bin].mean()
            confidence = confidences[in_bin].mean()
            ece += abs(accuracy - confidence) * in_bin.mean()
    return ece
```

### TAUS (Topology-Aware Uncertainty Score)
```python
def compute_taus(uncertainty, complexity):
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(
        uncertainty.flatten(),
        complexity.flatten()
    )
    return correlation
```

## Known Sources of Variation

### Deterministic vs Non-Deterministic
Even with fixed seeds, small variations (<0.5% DSC) can occur due to:
- CUDA non-deterministic operations
- Floating-point precision differences
- Batch normalization running statistics

### Platform Differences
Results may vary slightly across:
- Different GPU architectures (V100 vs A100)
- Different CUDA versions
- Different PyTorch versions

**Expected variation**: ±0.3% DSC, ±0.05 Betti error

## Validation Protocol

### Cross-Validation
We report mean ± std across 5-fold CV for ACDC:
- Fold 1: Cases 1-20
- Fold 2: Cases 21-40
- Fold 3: Cases 41-60
- Fold 4: Cases 61-80
- Fold 5: Cases 81-100

### Test Set Evaluation
- Synapse: Official test split (6 cases)
- BTCV: Cases 41-50
- ACDC: Cases 91-100

## Reproducing Paper Results

### Step 1: Data Preparation
```bash
python scripts/prepare_synapse.py --data_dir /path/to/Synapse
```

### Step 2: Training
```bash
python train.py --config configs/synapse.yaml --seed 42
```

### Step 3: Evaluation
```bash
python evaluate.py --checkpoint saved_models/best_model.pth --split test
```

### Expected Results (Synapse)
```
Mean DSC: 89.3 ± 0.2%
Betti Error: 0.54 ± 0.08
ECE: 0.043 ± 0.005
TAUS: 0.72 ± 0.03
```

## Troubleshooting

### Lower DSC than reported
- Check data preprocessing (spacing, intensity normalization)
- Verify batch size and learning rate
- Ensure 1000 epochs completed
- Check for data leakage in train/val/test splits

### Higher Betti Error
- Simplified topology implementation vs full GUDHI
- Check persistence threshold (τ=3 voxels)
- Verify topology loss weight (λ₁=0.3)

### Poor Calibration (High ECE)
- Increase calibration loss weight (λ₃)
- Use temperature scaling post-hoc
- Verify MC dropout enabled (25 samples)


**We are committed to reproducibility and will help debug.**