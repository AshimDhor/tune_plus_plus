import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TopologyLoss(nn.Module):
    """Topology preservation loss - placeholder for full implementation"""
    
    def __init__(self):
        super().__init__()
        # TODO: Implement persistent homology, Betti numbers, critical points
        
    def forward(self, pred, target):
        # Placeholder - requires GUDHI integration
        return torch.tensor(0.0, device=pred.device, requires_grad=True)


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
        loss = (error / (2 * sigma_a ** 2) + torch.log(sigma_a ** 2 + 1e-5)).mean()
        return loss
    
    def epistemic_loss(self, pred_single, pred_mc):
        """KL divergence between single pass and MC dropout"""
        pred_single = F.softmax(pred_single, dim=1)
        pred_mc = F.softmax(pred_mc, dim=1)
        
        kl = (pred_single * torch.log((pred_single + 1e-10) / (pred_mc + 1e-10))).sum(dim=1).mean()
        return kl
    
    def alignment_loss(self, sigma_total, complexity_score):
        """Align uncertainty with topological complexity"""
        return F.mse_loss(sigma_total, complexity_score)
    
    def compute_complexity_score(self, pred):
        """Compute topological complexity - simplified version"""
        # TODO: Replace with actual boundary, junction, anomaly detection
        # This is a placeholder using gradients as proxy for boundaries
        grad_x = torch.abs(pred[:, :, :-1, :, :] - pred[:, :, 1:, :, :])
        grad_y = torch.abs(pred[:, :, :, :-1, :] - pred[:, :, :, 1:, :])
        grad_z = torch.abs(pred[:, :, :, :, :-1] - pred[:, :, :, :, 1:])
        
        # Pad to match original size
        grad_x = F.pad(grad_x, (0, 0, 0, 0, 0, 1))
        grad_y = F.pad(grad_y, (0, 0, 0, 1, 0, 0))
        grad_z = F.pad(grad_z, (0, 1, 0, 0, 0, 0))
        
        complexity = (grad_x + grad_y + grad_z).mean(dim=1, keepdim=True)
        return complexity
    
    def forward(self, pred, target, sigma_a, sigma_e, pred_mc=None):
        loss_aleatoric = self.aleatoric_loss(pred, target, sigma_a)
        
        loss_epistemic = torch.tensor(0.0, device=pred.device)
        if pred_mc is not None:
            loss_epistemic = self.epistemic_loss(pred, pred_mc)
        
        sigma_total = sigma_a + sigma_e
        complexity = self.compute_complexity_score(pred)
        loss_align = self.alignment_loss(sigma_total, complexity)
        
        return loss_aleatoric + loss_epistemic + 0.5 * loss_align


class CalibrationLoss(nn.Module):
    """ECE + Brier Score"""
    
    def __init__(self, n_bins=10):
        super().__init__()
        self.n_bins = n_bins
        
    def forward(self, pred, target):
        # Brier score
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 4, 1, 2, 3).float()
        brier = ((pred_soft - target_one_hot) ** 2).mean()
        
        # ECE - simplified version
        ece = torch.tensor(0.0, device=pred.device)
        
        return ece + 0.5 * brier


class HierarchicalLoss(nn.Module):
    """Multi-scale topology consistency"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, features_list):
        # TODO: Implement persistence diagram comparison across scales
        return torch.tensor(0.0, device=features_list[0].device, requires_grad=True)


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
        
        # Segmentation loss
        loss_seg = self.dice_loss(pred, target) + self.ce_loss(pred, target)
        
        # Topology loss
        loss_topo = self.topo_loss(pred, target)
        
        # Uncertainty loss
        loss_unc = self.unc_loss(pred, target, sigma_a, sigma_e)
        
        # Calibration loss
        loss_calib = self.calib_loss(pred, target)
        
        # Hierarchical loss
        loss_hier = self.hier_loss(outputs.get('uncertainties', []))
        
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