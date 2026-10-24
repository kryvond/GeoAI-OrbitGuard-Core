import torch
import torch.nn as nn
import torch.nn.functional as F
from kryvond.pipeline.spectral_indices import GPUSpectralProcessor


class SpectralChannelAttention(nn.Module):
    """Squeeze-and-Excitation attention across multi-spectral & SAR bands."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ResidualConvBlock(nn.Module):
    """Residual convolutional block with Spectral Channel Attention."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.05):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.SiLU(inplace=True)
        self.attn = SpectralChannelAttention(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.block(x)
        return self.attn(self.act(out + residual))


class SpatialSelfAttentionBottleneck(nn.Module):
    """Vision Transformer (ViT) Multi-Head Self-Attention bottleneck for global swath context."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(B, C, H, W)


class TerraVisionGeoModel(nn.Module):
    """
    Kryvond TerraVision — Multi-Spectral + SAR Semantic Segmentation Architecture.
    Classes:
      0: Barren / Pasture / Background
      1: Critical Infrastructure & Urban Built-up
      2: Surface Water & Active Flood Inundation
      3: High-Density Vegetation & Agricultural Canopy
      4: Industrial & Energy Grid Assets
    """
    def __init__(self, in_bands: int = 12, num_classes: int = 5, base_dim: int = 32):
        super().__init__()
        self.spectral_frontend = GPUSpectralProcessor()
        total_in = in_bands + 4  # 12 raw bands + 4 GPU-computed physical indices

        self.spectral_skip = nn.Sequential(
            nn.Conv2d(total_in, base_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.SiLU(inplace=True),
        )

        self.enc1 = ResidualConvBlock(total_in, base_dim)
        self.enc2 = ResidualConvBlock(base_dim, base_dim * 2)
        self.enc3 = ResidualConvBlock(base_dim * 2, base_dim * 4)

        self.pool = nn.MaxPool2d(2)
        self.bottleneck = SpatialSelfAttentionBottleneck(dim=base_dim * 4, num_heads=4)

        self.up2 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualConvBlock(base_dim * 4, base_dim * 2)

        self.up1 = nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=2, stride=2)
        self.dec1 = ResidualConvBlock(base_dim * 2, base_dim)

        self.head = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(base_dim, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.spectral_frontend(x)
        s0 = self.spectral_skip(feats)

        e1 = self.enc1(feats)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(e3)

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        logits = self.head(torch.cat([d1, s0], dim=1))
        probs = F.softmax(logits, dim=1)
        return {"logits": logits, "probs": probs}


class HybridSegmentationLoss(nn.Module):
    """Combined Cross-Entropy + Multi-Class Soft Dice Loss."""
    def __init__(self, num_classes: int = 5, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - torch.mean(dice_score)
        return 0.6 * ce_loss + 0.4 * dice_loss
