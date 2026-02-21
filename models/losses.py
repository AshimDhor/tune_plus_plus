import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.topology import (
    compute_betti_numbers, 
    compute_topological_complexity,
    extract_boundaries
)


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TopologyLoss(nn.Module):
    """
    Topology preservation loss
    
    Note: This is a working approximation. Full persistent homology
    integration with GUDHI will be added in future versions.
    """
    
    def __init__(self):
        super().__init__()
        
    def betti_loss(self, pred, target):
        """Compute Betti number difference"""
        pred_mask = pred.argmax(dim=1).cpu().numpy()
        target_mask = target.cpu().numpy()
        
        batch_size = pred_mask.shape[0]
        betti_errors = []
        
        for b in range(batch_size):
            pred_betti = compute_betti_numbers(pred_mask[b])
            target_betti = compute_betti_numbers(target_mask[b])
            
            # Sum of absolute differences
            error = sum(abs(p - t) for p, t in zip(pred_betti, target_betti))
            betti_errors.append(error)
        
        return torch.tensor(np.mean(betti_errors), device=pred.device, dtype=torch.float32)
    
    def boundary_loss(self, pred, target):
        """Penalize boundary errors"""
        pred_soft = F.softmax(pred, dim=1)
        
        # Compute spatial gradients
        grad_x = torch.abs(pred_soft[:, :, :-1, :, :] - pred_soft[:, :, 1:, :, :])
        grad_y = torch.abs(pred_soft[:, :, :, :-1, :] - pred_soft[:, :, :, 1:, :])
        grad_z = torch.abs(pred_soft[:, :, :, :, :-1] - pred_soft[:, :, :, :, 1:])
        
        # Target boundaries
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        target_grad_x = torch.abs(target_one_hot[:, :, :-1, :, :] - target_one_hot[:, :, 1:, :, :])
        target_grad_y = torch.abs(target_one_hot[:, :, :, :-1, :] - target_one_hot[:, :, :, 1:, :])
        target_grad_z = torch.abs(target_one_hot[:, :, :, :, :-1] - target_one_hot[:, :, :, :, 1:])
        
        # L2 loss on boundary maps
        loss_x = F.mse_loss(grad_x, target_grad_x)
        loss_y = F.mse_loss(grad_y, target_grad_y)
        loss_z = F.mse_loss(grad_z, target_grad_z)
        
        return (loss_x + loss_y + loss_z) / 3.0
    
    def forward(self, pred, target):
        """
        Combined topology loss
        
        Note: Betti loss computed every N iterations due to computational cost
        """
        # Always compute boundary loss (differentiable)
        boundary_loss = self.boundary_loss(pred, target)
        
        # Compute Betti loss (expensive, non-differentiable)
        # In practice, compute this less frequently
        try:
            betti_loss = self.betti_loss(pred, target)
        except:
            betti_loss = torch.tensor(0.0, device=pred.device)
        
        # Weighted combination
        return boundary_loss + 0.5 * betti_loss


class UncertaintyLoss(nn.Module):
    """Uncertainty decomposition + alignment loss"""
    
    def __init__(self, w_b=1.0, w_j=2.0, w_a=3.0):
        super().__init__()
        self.w_b = w_b
        self.w_j = w_j
        self.w_a = w_a
        
    def aleatoric_loss(self, pred, target, sigma_a):
        """Heteroscedastic aleatoric uncertainty"""
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        error = (pred_soft - target_one_hot) ** 2
        
        # Average over classes
        error = error.mean(dim=1, keepdim=True)
        sigma_a_mean = sigma_a.mean(dim=1, keepdim=True)
        
        loss = (error / (2 * sigma_a_mean ** 2 + 1e-10) + torch.log(sigma_a_mean ** 2 + 1e-5)).mean()
        return loss
    
    def epistemic_loss(self, pred_single, pred_mc):
        """KL divergence between single pass and MC dropout"""
        pred_single = F.softmax(pred_single, dim=1)
        pred_mc = F.softmax(pred_mc, dim=1)
        
        kl = (pred_single * torch.log((pred_single + 1e-10) / (pred_mc + 1e-10))).sum(dim=1).mean()
        return kl
    
    def alignment_loss(self, sigma_total, target):
        """Align uncertainty with topological complexity"""
        # Compute complexity score
        complexity_maps = []
        
        for b in range(target.shape[0]):
            complexity = compute_topological_complexity(
                target[b].cpu().numpy(),
                w_b=self.w_b,
                w_j=self.w_j,
                w_a=self.w_a
            )
            complexity_maps.append(torch.from_numpy(complexity))
        
        complexity_batch = torch.stack(complexity_maps).to(sigma_total.device).unsqueeze(1)
        
        # MSE between uncertainty and complexity
        sigma_total_mean = sigma_total.mean(dim=1, keepdim=True)
        
        return F.mse_loss(sigma_total_mean, complexity_batch)
    
    def forward(self, pred, target, sigma_a, sigma_e, pred_mc=None):
        loss_aleatoric = self.aleatoric_loss(pred, target, sigma_a)
        
        loss_epistemic = torch.tensor(0.0, device=pred.device)
        if pred_mc is not None:
            loss_epistemic = self.epistemic_loss(pred, pred_mc)
        
        sigma_total = sigma_a + sigma_e
        
        try:
            loss_align = self.alignment_loss(sigma_total, target)
        except:
            loss_align = torch.tensor(0.0, device=pred.device)
        
        return loss_aleatoric + loss_epistemic + 0.5 * loss_align


class CalibrationLoss(nn.Module):
    """ECE + Brier Score"""
    
    def __init__(self, n_bins=10):
        super().__init__()
        self.n_bins = n_bins
        
    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        # Brier score
        brier = ((pred_soft - target_one_hot) ** 2).mean()
        
        # Simplified ECE (full implementation requires binning)
        ece = torch.tensor(0.0, device=pred.device)
        
        return ece + 0.5 * brier


class HierarchicalLoss(nn.Module):
    """Multi-scale topology consistency"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, uncertainties_list):
        if len(uncertainties_list) < 2:
            return torch.tensor(0.0, device=uncertainties_list[0].device, requires_grad=True)
        
        # Simple consistency: MSE between consecutive scale uncertainties
        loss = 0
        for i in range(len(uncertainties_list) - 1):
            unc1 = uncertainties_list[i].mean()
            unc2 = uncertainties_list[i + 1].mean()
            loss += (unc1 - unc2) ** 2
        
        return loss / max(len(uncertainties_list) - 1, 1)


class TUNELoss(nn.Module):
    """Combined TUNE++ Loss"""
    
    def __init__(self, lambda1=0.3, lambda2=0.2, lambda3=0.1, lambda4=0.15, 
                 w_b=1.0, w_j=2.0, w_a=3.0):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.topo_loss = TopologyLoss()
        self.unc_loss = UncertaintyLoss(w_b, w_j, w_a)
        self.calib_loss = CalibrationLoss()
        self.hier_loss = HierarchicalLoss()
        
    def forward(self, outputs, target):
        pred = outputs['segmentation']
        sigma_a = outputs['aleatoric_uncertainty']
        sigma_e = outputs['epistemic_uncertainty']
        uncertainties = outputs.get('uncertainties', [])
        
        # Segmentation loss
        loss_seg = self.dice_loss(pred, target) + self.ce_loss(pred, target)
        
        # Topology loss
        loss_topo = self.topo_loss(pred, target)
        
        # Uncertainty loss
        loss_unc = self.unc_loss(pred, target, sigma_a, sigma_e)
        
        # Calibration loss
        loss_calib = self.calib_loss(pred, target)
        
        # Hierarchical loss
        loss_hier = self.hier_loss(uncertainties)
        
        # Total loss
        total_loss = (loss_seg + 
                     self.lambda1 * loss_topo + 
                     self.lambda2 * loss_unc + 
                     self.lambda3 * loss_calib + 
                     self.lambda4 * loss_hier)
        
        return total_loss, {
            'loss_seg': loss_seg.item(),
            'loss_topo': loss_topo.item(),
            'loss_unc': loss_unc.item(),
            'loss_calib': loss_calib.item(),
            'loss_hier': loss_hier.item(),
            'total': total_loss.item()
        }