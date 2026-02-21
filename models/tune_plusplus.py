import torch
import torch.nn as nn
import torch.nn.functional as F
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
        return x, (H, W, D)


class DownsampleLayer(nn.Module):
    """Downsample with convolution"""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm3d(out_dim)
        
    def forward(self, x):
        return self.norm(self.conv(x))


class UpsampleLayer(nn.Module):
    """Upsample with transposed convolution"""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_dim, out_dim, kernel_size=2, stride=2)
        self.norm = nn.BatchNorm3d(out_dim)
        
    def forward(self, x):
        return self.norm(self.conv(x))


class EncoderStage(nn.Module):
    """Encoder stage with TUPA blocks"""
    
    def __init__(self, dim, depth=3, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            TUPABlock(dim, num_heads) for _ in range(depth)
        ])
        
    def forward(self, x):
        uncertainties = []
        for blk in self.blocks:
            x, unc = blk(x)
            uncertainties.append(unc)
        return x, uncertainties


class DecoderStage(nn.Module):
    """Decoder stage with skip connections"""
    
    def __init__(self, dim, depth=3, num_heads=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            TUPABlock(dim, num_heads) for _ in range(depth)
        ])
        
    def forward(self, x, skip):
        x = x + skip
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
        
        self.num_stages = len(depths)
        dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(in_channels, embed_dim)
        
        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i in range(self.num_stages):
            self.encoder_stages.append(
                EncoderStage(dims[i], depths[i], num_heads[i])
            )
            if i < self.num_stages - 1:
                self.downsample_layers.append(
                    DownsampleLayer(dims[i], dims[i + 1])
                )
        
        # Decoder stages
        self.upsample_layers = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        
        for i in range(self.num_stages - 1, 0, -1):
            self.upsample_layers.append(
                UpsampleLayer(dims[i], dims[i - 1])
            )
            self.decoder_stages.append(
                DecoderStage(dims[i - 1], depths[i - 1], num_heads[i - 1])
            )
        
        # Output heads
        self.seg_head = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim, out_channels, 1),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)  # Match patch size
        )
        
        self.aleatoric_head = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim // 2, out_channels, 1),
            nn.Softplus(),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        )
        
        self.epistemic_head = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim // 2, out_channels, 1),
            nn.Softplus(),
            nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
        )
        
    def forward(self, x):
        # Patch embedding
        x, spatial_dims = self.patch_embed(x)
        
        # Encoder
        encoder_features = []
        all_uncertainties = []
        
        for i, encoder in enumerate(self.encoder_stages):
            x, unc = encoder(x)
            encoder_features.append(x)
            all_uncertainties.extend(unc)
            
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)
        
        # Decoder
        for i, (upsample, decoder) in enumerate(zip(self.upsample_layers, self.decoder_stages)):
            x = upsample(x)
            skip = encoder_features[-(i + 2)]  # Get corresponding encoder feature
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