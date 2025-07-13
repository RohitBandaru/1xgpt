import json
from dataclasses import dataclass
# from typing import Optional


@dataclass
class VJEPAConfig:
    """Configuration for V-JEPA2-AC based world model"""
    
    # V-JEPA2 backbone config
    model_name: str = "vit_ac_giant"  # vit_ac_giant, vit_ac_large
    img_size: int = 256
    patch_size: int = 16
    tubelet_size: int = 2
    num_frames: int = 16
    pretrained: bool = True
    
    # Predictor config (from V-JEPA2-AC)
    pred_depth: int = 24
    pred_num_heads: int = 16
    pred_embed_dim: int = 1024
    pred_is_frame_causal: bool = True
    use_extrinsics: bool = False
    
    # 1X specific config
    T: int = 16  # temporal sequence length (frames)
    S: int = 256  # spatial sequence length (16x16 = 256)
    image_vocab_size: int = 262144  # 2^18 MAGVIT2 vocab
    
    # Factorization config (same as GENIE)
    num_factored_vocabs: int = 2
    factored_vocab_size: int = 512  # 512^2 = 262144
    
    # Action config for 1X dataset
    action_vocab_size: int = 65536  # uint16 action tokens
    action_embed_dim: int = 256  # embedding dimension for scalar actions
    
    # Training config
    loss_type: str = "cross_entropy"  # "cross_entropy", "l1"
    l1_loss_weight: float = 1.0
    token_loss_weight: float = 1.0
    freeze_backbone: bool = True  # Freeze V-JEPA encoder, train only predictor
    
    def save_pretrained(self, json_path):
        with open(json_path, "w") as f:
            json.dump(vars(self), f, indent=2)

    @classmethod
    def from_pretrained(cls, json_path):
        with open(json_path, "r") as f:
            config = json.load(f)
        return cls(**config)

    def shallow_copy(self):
        return VJEPAConfig(**vars(self))