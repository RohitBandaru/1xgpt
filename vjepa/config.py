import json
from dataclasses import dataclass
from transformers import PretrainedConfig
# from typing import Optional


class VJEPAConfig(PretrainedConfig):
    """Configuration for V-JEPA2-AC based world model"""
    
    def __init__(
        self,
        # V-JEPA2 backbone config
        model_name: str = "vit_ac_giant",  # vit_ac_giant, vit_ac_large
        img_size: int = 256,
        patch_size: int = 16,
        tubelet_size: int = 2,
        num_frames: int = 16,
        pretrained: bool = True,
        
        # Predictor config (from V-JEPA2-AC)
        pred_depth: int = 24,
        pred_num_heads: int = 16,
        pred_embed_dim: int = 1024,
        pred_is_frame_causal: bool = True,
        use_extrinsics: bool = False,
        
        # 1X specific config
        T: int = 16,  # temporal sequence length (frames)
        S: int = 256,  # spatial sequence length (16x16 = 256)
        image_vocab_size: int = 262144,  # 2^18 MAGVIT2 vocab
        
        # Factorization config (same as GENIE)
        num_factored_vocabs: int = 2,
        factored_vocab_size: int = 512,  # 512^2 = 262144
        
        # Action config for 1X dataset
        action_vocab_size: int = 65536,  # uint16 action tokens
        action_embed_dim: int = 256,  # embedding dimension for scalar actions
        
        # Training config
        loss_type: str = "cross_entropy",  # "cross_entropy", "l1"
        l1_loss_weight: float = 1.0,
        token_loss_weight: float = 1.0,
        freeze_backbone: bool = True,  # Freeze V-JEPA encoder, train only predictor
        
        # MaskGIT training config
        max_corrupt_rate: float = 0.2,  # Corrupt all tokens, uniform between [0, max_corrupt_rate]
        non_mlm_ratio: float = 0.5,
        num_prompt_frames: int = 8,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # V-JEPA2 backbone config
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.pretrained = pretrained
        
        # Predictor config (from V-JEPA2-AC)
        self.pred_depth = pred_depth
        self.pred_num_heads = pred_num_heads
        self.pred_embed_dim = pred_embed_dim
        self.pred_is_frame_causal = pred_is_frame_causal
        self.use_extrinsics = use_extrinsics
        
        # 1X specific config
        self.T = T
        self.S = S
        self.image_vocab_size = image_vocab_size
        
        # Factorization config (same as GENIE)
        self.num_factored_vocabs = num_factored_vocabs
        self.factored_vocab_size = factored_vocab_size
        
        # Action config for 1X dataset
        self.action_vocab_size = action_vocab_size
        self.action_embed_dim = action_embed_dim
        
        # Training config
        self.loss_type = loss_type
        self.l1_loss_weight = l1_loss_weight
        self.token_loss_weight = token_loss_weight
        self.freeze_backbone = freeze_backbone
        
        # MaskGIT training config
        self.max_corrupt_rate = max_corrupt_rate
        self.non_mlm_ratio = non_mlm_ratio
        self.num_prompt_frames = num_prompt_frames
    
    def shallow_copy(self):
        return VJEPAConfig(**self.to_dict())