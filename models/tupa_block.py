import torch
import torch.nn as nn
import torch.nn.functional as F


class TUPABlock(nn.Module):
    """Topology-Uncertainty Aware Paired Attention Block"""
    
    def __init__(self, dim, num_heads=8, proj_dim=64, lambda_t=0.3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.proj_dim = proj_dim
        self.lambda_t = lambda_t
        
        # Shared query-key projection
        self.qk_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        
        # Spatial attention with dimensionality reduction
        self.spatial_proj = nn.Linear(dim, proj_dim)
        self.v_spatial = nn.Linear(dim, dim)
        
        # Channel attention
        self.v_channel = nn.Linear(dim, dim)
        
        # Topology-aware attention
        self.critical_point_detector = nn.Sequential(
            nn.Conv3d(dim, dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim // 2, dim // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim // 4, 1, 1)
        )
        self.v_topology = nn.Linear(dim, dim)
        
        # Uncertainty estimation
        self.uncertainty_mlp = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 2),  # aleatoric + epistemic_init
            nn.Softplus()
        )
        
        # Convolutional refinement
        self.refine = nn.Sequential(
            nn.Conv3d(dim, dim, 3, padding=1),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim, dim, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W, D) input features
        Returns:
            x_out: (B, C, H, W, D) refined features
            uncertainty: (B, 2, H, W, D) aleatoric + epistemic
        """
        B, C, H, W, D = x.shape
        N = H * W * D
        
        # Flatten to tokens
        x_flat = x.flatten(2).transpose(1, 2)  # B, N, C
        x_norm = self.norm(x_flat)
        
        # Shared Q, K
        qk = self.qk_proj(x_norm)  # B, N, C
        
        # Spatial attention with efficient projection
        k_proj = self.spatial_proj(qk.transpose(1, 2)).transpose(1, 2)  # B, P, C
        v_spatial = self.v_spatial(x_norm)
        v_spatial_proj = self.spatial_proj(v_spatial.transpose(1, 2)).transpose(1, 2)
        
        attn_s = F.softmax(torch.matmul(qk, k_proj.transpose(1, 2)) / (C ** 0.5), dim=-1)
        x_spatial = torch.matmul(attn_s, v_spatial_proj)
        
        # Channel attention
        v_channel = self.v_channel(x_norm)
        attn_c = F.softmax(torch.matmul(qk.transpose(1, 2), qk) / (C ** 0.5), dim=-1)
        x_channel = torch.matmul(v_channel, attn_c)
        
        # Topology attention
        topo_mask = self.critical_point_detector(x).flatten(2)  # B, 1, N
        v_topo = self.v_topology(x_norm)
        
        attn_t = F.softmax(
            torch.matmul(qk, qk.transpose(1, 2)) / (C ** 0.5) + self.lambda_t * topo_mask.unsqueeze(-1),
            dim=-1
        )
        x_topo = torch.matmul(attn_t, v_topo)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_mlp(x_norm)  # B, N, 2
        sigma_total = uncertainty.sum(dim=-1, keepdim=True)  # B, N, 1
        
        # Adaptive fusion weights
        w_s = torch.sigmoid(-sigma_total)
        w_t = torch.sigmoid(sigma_total)
        
        # Fuse branches
        x_fused = w_s * (x_spatial + x_channel) + w_t * x_topo
        
        # Reshape back to 3D
        x_fused = x_fused.transpose(1, 2).reshape(B, C, H, W, D)
        
        # Refinement with residual
        x_out = self.refine(x_fused) + x
        
        # Reshape uncertainty
        uncertainty = uncertainty.transpose(1, 2).reshape(B, 2, H, W, D)
        
        return x_out, uncertainty