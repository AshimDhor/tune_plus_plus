import torch
import torch.nn as nn
from .tupa_block import TUPABlock


class PatchEmbedding(nn.Module):
    """Convert 3D volume into tokens"""
    
    def __init__(self, in_channels=1, embed_dim=32, patch_size=(4, 4, 2)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.proj(x)  # B, C, H, W, D
        B, C, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x, (H, W, D)


class EncoderStage(nn.Module):
    """Single encoder stage with TUPA"""
    
    def __init__(self, dim, depth=3, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            TUPABlock(dim, num_heads) for _ in range(depth)
        ])
        self.downsample = nn.Conv3d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x, spatial_dims):
        H, W, D = spatial_dims
        B, N, C = x.shape
        
        # Reshape to 3D
        x = x.transpose(1, 2).reshape(B, C, H, W, D)
        
        uncertainties = []
        for blk in self.blocks:
            x, unc = blk(x)
            uncertainties.append(unc)
        
        # Downsample
        x_down = self.downsample(x)
        
        return x, x_down, uncertainties


class DecoderStage(nn.Module):
    """Single decoder stage with skip connections"""
    
    def __init__(self, dim, depth=3, num_heads=8):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(dim, dim // 2, kernel_size=2, stride=2)
        self.blocks = nn.ModuleList([
            TUPABlock(dim // 2, num_heads) for _ in range(depth)
        ])
        
    def forward(self, x, skip):
        x = self.upsample(x)
        x = x + skip  # Skip connection
        
        uncertainties = []
        for blk in self.blocks:
            x, unc = blk(x)
            uncertainties.append(unc)
            
        return x, uncertainties


class TUNEPlusPlus(nn.Module):
    """TUNE++ Main Architecture"""
    
    def __init__(self, in_channels=1, out_channels=9, img_size=(96, 96, 96), 
                 embed_dim=32, depths=[3, 3, 9, 3], num_heads=[3, 6, 12, 24]):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, embed_dim)
        
        # Encoder
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.encoders = nn.ModuleList([
            EncoderStage(dims[i], depths[i], num_heads[i]) 
            for i in range(len(depths))
        ])
        
        # Decoder
        self.decoders = nn.ModuleList([
            DecoderStage(dims[i], depths[i], num_heads[i])
            for i in range(len(depths) - 1, 0, -1)
        ])
        
        # Output heads
        self.seg_head = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, 3, padding=1),
            nn.Conv3d(embed_dim, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )
        
        self.aleatoric_head = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim // 2, out_channels, 1),
            nn.Softplus()
        )
        
        self.epistemic_head = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim // 2, out_channels, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        # Embedding
        x, spatial_dims = self.patch_embed(x)
        
        # Encoder
        skips = []
        all_uncertainties = []
        
        for encoder in self.encoders:
            skip, x, unc = encoder(x, spatial_dims)
            skips.append(skip)
            all_uncertainties.extend(unc)
            # Update spatial dims after downsampling
            spatial_dims = tuple(d // 2 for d in spatial_dims)
        
        # Decoder
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            x, unc = decoder(x, skip)
            all_uncertainties.extend(unc)
        
        # Output heads
        segmentation = self.seg_head(x)
        aleatoric = self.aleatoric_head(x)
        epistemic = self.epistemic_head(x)
        
        return {
            'segmentation': segmentation,
            'aleatoric_uncertainty': aleatoric,
            'epistemic_uncertainty': epistemic,
            'uncertainties': all_uncertainties
        }