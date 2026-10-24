import torch
import torch.nn as nn


class GPUSpectralProcessor(nn.Module):
    """
    GPU-Accelerated Multi-Spectral & SAR Tensor Preprocessor.
    Ingests 12-band Sentinel-2 optical + Dual-Pol Sentinel-1 SAR tensors (B, 12, H, W)
    and computes normalized radiometric indices directly on CUDA/CPU devices.

    Band Mapping (0-indexed):
      B01 (Coastal): 0 | B02 (Blue): 1  | B03 (Green): 2 | B04 (Red): 3
      B05 (RE1): 4     | B06 (RE2): 5   | B07 (RE3): 6   | B08 (NIR): 7
      B11 (SWIR1): 8   | B12 (SWIR2): 9 | SAR VV: 10     | SAR VH: 11
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def compute_indices(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Computes NDVI, NDWI, NDBI, and SAR Polarization Ratio."""
        green = x[:, 2:3, :, :]
        red = x[:, 3:4, :, :]
        nir = x[:, 7:8, :, :]
        swir1 = x[:, 8:9, :, :]
        vv = x[:, 10:11, :, :]
        vh = x[:, 11:12, :, :]

        ndvi = torch.clamp((nir - red) / (nir + red + self.eps), -1.0, 1.0)
        ndwi = torch.clamp((green - nir) / (green + nir + self.eps), -1.0, 1.0)
        ndbi = torch.clamp((swir1 - nir) / (swir1 + nir + self.eps), -1.0, 1.0)
        sar_ratio = torch.clamp((vv - vh) / (vv + vh + self.eps), -1.0, 1.0)

        return {
            "ndvi": ndvi,
            "ndwi": ndwi,
            "ndbi": ndbi,
            "sar_ratio": sar_ratio,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Appends 4 physical remote-sensing indices to the 12 raw bands -> 16 channels."""
        green = x[:, 2:3, :, :]
        red = x[:, 3:4, :, :]
        nir = x[:, 7:8, :, :]
        swir1 = x[:, 8:9, :, :]
        vv = x[:, 10:11, :, :]
        vh = x[:, 11:12, :, :]

        ndvi = torch.clamp((nir - red) / (nir + red + self.eps), -1.0, 1.0)
        ndwi = torch.clamp((green - nir) / (green + nir + self.eps), -1.0, 1.0)
        ndbi = torch.clamp((swir1 - nir) / (swir1 + nir + self.eps), -1.0, 1.0)
        sar_ratio = torch.clamp((vv - vh) / (vv + vh + self.eps), -1.0, 1.0)
        return torch.cat([x, ndvi, ndwi, ndbi, sar_ratio], dim=1)
