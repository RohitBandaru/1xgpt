"""
V-JEPA Encoder for Continuous Embeddings

Loads V-JEPA2-AC backbone and outputs continuous ViT-G embeddings.
Preserves rich representations without quantization for predictor training.

Note: For discrete tokens, use precomputed COSMOS tokens from 1X dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np
from pathlib import Path

from .config import VJEPAEncoderConfig



class VJEPAEncoder(nn.Module):
    """
    V-JEPA Encoder for continuous embeddings.
    
    Loads pretrained ViT-Giant backbone and outputs continuous embeddings
    that preserve the rich representations from the pretrained model.
    
    For discrete tokens, use precomputed COSMOS tokens from 1X dataset.
    """
    
    def __init__(self, config: VJEPAEncoderConfig):
        super().__init__()
        self.config = config
        
        # Load V-JEPA2-AC encoder
        self._load_vjepa_encoder()
    
    def _load_vjepa_encoder(self):
        """Load V-JEPA2-AC encoder from PyTorch Hub"""
        print("Loading V-JEPA2-AC encoder...")
        
        if self.config.pretrained:
            # Load pretrained V-JEPA encoder
            encoder, _ = torch.hub.load(
                'facebookresearch/vjepa2',
                'vjepa2_ac_vit_giant',
                pretrained=True,
                verbose=False
            )
            self.encoder = encoder
            
            print(f"✅ Loaded pretrained V-JEPA encoder:")
            print(f"   Model: ViT-Giant (1.7B parameters)")
            print(f"   Embed dim: {self.encoder.embed_dim}")
            print(f"   Parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
        else:
            # Load architecture without pretrained weights
            encoder, _ = torch.hub.load(
                'facebookresearch/vjepa2',
                'vjepa2_ac_vit_giant',
                pretrained=False,
                verbose=False
            )
            self.encoder = encoder
            
            print(f"✅ Loaded V-JEPA encoder architecture (no pretrained weights):")
            print(f"   Model: ViT-Giant (1.7B parameters)")
            print(f"   Embed dim: {self.encoder.embed_dim}")
            print(f"   Parameters: {sum(p.numel() for p in self.encoder.parameters()):,}")
        
        # Freeze encoder (it's pretrained and used for feature extraction only)
        self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder parameters"""
        print("Freezing V-JEPA encoder...")
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Print parameter summary
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Parameter summary:")
        print(f"  Total: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  All parameters frozen (encoder used for feature extraction only)")
    
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
    
    def get_embedding_info(self) -> dict:
        """
        Get information about the continuous embeddings format.
        
        Returns:
            info: Dict with embedding dimensions and format details
        """
        return {
            'embed_dim': self.encoder.embed_dim,
            'model_type': 'ViT-Giant',
            'spatial_resolution': '16x16',  # Standard for 256x256 input
            'temporal_patches': 'varies',  # Depends on input video length
            'feature_type': 'continuous_embeddings',
            'quantized': False
        }
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass: raw video -> continuous embeddings.
        
        Args:
            pixel_values: [B, T, C, H, W] raw video frames
            
        Returns:
            continuous embeddings [B, N_patches, embed_dim]
        """
        return self.encode_video(pixel_values)
    
    def save_embeddings(self, embeddings: torch.FloatTensor, output_path: Union[str, Path], metadata: dict = None):
        """
        Save continuous V-JEPA embeddings to disk.
        
        Args:
            embeddings: [N_sequences, N_patches, embed_dim] V-JEPA embeddings
            output_path: Path to save embeddings
            metadata: Optional metadata dict
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as PyTorch tensor for full precision
        torch.save({
            'embeddings': embeddings.cpu(),
            'metadata': metadata or {},
            'format': 'continuous_embeddings',
            'embed_dim': embeddings.shape[-1],
            'dtype': str(embeddings.dtype)
        }, output_path)
        
        print(f"✅ Saved {embeddings.shape[0]} sequences ({embeddings.shape[1]} patches, {embeddings.shape[2]}D) to {output_path}")
    
    @classmethod
    def load_embeddings(cls, filepath: Union[str, Path]) -> Tuple[torch.FloatTensor, dict]:
        """
        Load continuous embeddings from disk.
        
        Args:
            filepath: Path to embeddings file
            
        Returns:
            embeddings: [N_sequences, N_patches, embed_dim] V-JEPA embeddings
            metadata: Metadata dict
        """
        data = torch.load(filepath, map_location='cpu')
        return data['embeddings'], data['metadata']
    
    def analyze_embeddings(self, pixel_values: torch.FloatTensor) -> dict:
        """
        Analyze continuous embedding quality and statistics.
        
        Args:
            pixel_values: [B, T, C, H, W] raw video frames
            
        Returns:
            analysis: Dict with embedding statistics
        """
        with torch.no_grad():
            # Get embeddings
            embeddings = self.encode_video(pixel_values)
            
            # Compute statistics
            analysis = {
                'input_shape': list(pixel_values.shape),
                'embeddings_shape': list(embeddings.shape),
                'embed_dim': embeddings.shape[-1],
                'num_patches': embeddings.shape[1],
                'embedding_stats': {
                    'mean': embeddings.mean().item(),
                    'std': embeddings.std().item(),
                    'min': embeddings.min().item(),
                    'max': embeddings.max().item(),
                    'norm_mean': embeddings.norm(dim=-1).mean().item(),
                    'norm_std': embeddings.norm(dim=-1).std().item()
                }
            }
        
        return analysis


def create_vjepa_encoder(config_path: str) -> VJEPAEncoder:
    """
    Factory function to create V-JEPA encoder for continuous embeddings.
    
    Args:
        config_path: Path to V-JEPA config file
        
    Returns:
        VJEPAEncoder instance
    """
    from .config import VJEPAEncoderConfig
    
    config = VJEPAEncoderConfig.from_pretrained(config_path)
    encoder = VJEPAEncoder(config)
    
    return encoder