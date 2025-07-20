"""
V-JEPA Encoder for Continuous Embeddings

Loads V-JEPA2-AC backbone and outputs continuous ViT-G embeddings.
Preserves rich representations without quantization for predictor training.

Note: For discrete tokens, use precomputed COSMOS tokens from 1X dataset.
"""

from typing import Optional

import torch
import torch.nn as nn

from .config import VJEPAEncoderConfig


class VJEPAEncoder(nn.Module):
    """V-JEPA Encoder for continuous embeddings."""

    def __init__(self, config: VJEPAEncoderConfig):
        super().__init__()
        self.config = config
        self._load_vjepa_encoder()

    def _load_vjepa_encoder(self):
        """Load V-JEPA2-AC encoder from PyTorch Hub"""
        encoder, _ = torch.hub.load(
            "facebookresearch/vjepa2",
            "vjepa2_ac_vit_giant",
            pretrained=self.config.pretrained,
            verbose=False,
        )
        self.encoder = encoder
        self._freeze_encoder()

    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def encode_video(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """
        Encode raw video to continuous V-JEPA embeddings.

        This returns the raw ViT-G embeddings without any quantization,
        preserving the rich continuous representations from the pretrained model.

        Args:
            pixel_values: [B, T, C, H, W] raw video frames

        Returns:
            embeddings: [B, N_patches, embed_dim] continuous V-JEPA embeddings
        """
        with torch.no_grad():  # Encoder is frozen
            embeddings = self.encoder(pixel_values)
        return embeddings

    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass: raw video -> continuous embeddings.

        Args:
            pixel_values: [B, T, C, H, W] raw video frames

        Returns:
            continuous embeddings [B, N_patches, embed_dim]
        """
        return self.encode_video(pixel_values)
