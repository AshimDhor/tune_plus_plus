import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr


def compute_dice(pred, target, num_classes):
    """
    Compute Dice score per class
    
    Args:
        pred: Predictions (B, C, H, W, D)
        target: Ground truth (B, H, W, D)
        num_classes: Number of classes
    Returns:
        dice_scores: Per-class Dice scores
    """
    pred = F.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
    
    dice_scores = []
    for c in range(num_classes):
        pred_c = pred[:, c]
        target_c = target_one_hot[:, c]
        
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_scores.append(dice.item())
    
    return np.array(dice_scores)


def compute_ece(pred, target, n_bins=10):
    """
    Compute Expected Calibration Error
    
    Args:
        pred: Predictions (B, C, H, W, D)
        target: Ground truth (B, H, W, D)
        n_bins: Number of bins
    Returns:
        ece: Expected calibration error
    """
    pred_soft = F.softmax(pred, dim=1)
    confidences, predictions = pred_soft.max(dim=1)
    
    confidences = confidences.flatten().cpu().numpy()
    predictions = predictions.flatten().cpu().numpy()
    target = target.flatten().cpu().numpy()
    
    correct = (predictions == target).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_taus(uncertainty, complexity_score):
    """
    Compute Topology-Aware Uncertainty Score (correlation)
    
    Args:
        uncertainty: Total uncertainty map
        complexity_score: Topological complexity score
    Returns:
        taus: Pearson correlation coefficient
    """
    unc_flat = uncertainty.flatten().cpu().numpy()
    comp_flat = complexity_score.flatten().cpu().numpy()
    
    correlation, _ = pearsonr(unc_flat, comp_flat)
    
    return correlation


def compute_betti_error(pred_mask, gt_mask):
    """
    Compute Betti number error
    
    Args:
        pred_mask: Predicted segmentation
        gt_mask: Ground truth segmentation
    Returns:
        betti_error: Sum of absolute differences in Betti numbers
    """
    from .topology import compute_betti_numbers
    
    pred_betti = compute_betti_numbers(pred_mask)
    gt_betti = compute_betti_numbers(gt_mask)
    
    betti_error = sum(abs(p - g) for p, g in zip(pred_betti, gt_betti))
    
    return betti_error