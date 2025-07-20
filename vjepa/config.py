import json
from dataclasses import dataclass

from transformers import PretrainedConfig

# from typing import Optional


class VJEPAEncoderConfig(PretrainedConfig):
    """Configuration for V-JEPA Encoder (ViT-G backbone for continuous embeddings)"""

    def __init__(
        self,
        pretrained: bool = True,
        vjepa_embed_dim: int = 1408,  # ViT-G embedding dimension
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pretrained = pretrained
        self.vjepa_embed_dim = vjepa_embed_dim

        # Validation
        if not isinstance(pretrained, bool):
            raise TypeError(f"pretrained must be bool, got {type(pretrained)}")
        if not isinstance(vjepa_embed_dim, int) or vjepa_embed_dim <= 0:
            raise ValueError(
                f"vjepa_embed_dim must be positive int, got {vjepa_embed_dim}"
            )

    def shallow_copy(self):
        return VJEPAEncoderConfig(**self.to_dict())


class VJEPAPredictorConfig(PretrainedConfig):
    """Configuration for V-JEPA Predictor"""

    def __init__(
        self,
        # Model config
        pretrained: bool = True,
        freeze_backbone: bool = True,
        # Data dimensions
        T: int = 16,  # temporal frames
        S: int = 256,  # spatial patches
        image_vocab_size: int = 262144,
        # Input mode
        input_mode: str = "discrete",  # "discrete" or "continuous"
        vjepa_embed_dim: int = 1408,  # ViT-G dimension
        # Factorization
        num_factored_vocabs: int = 2,
        factored_vocab_size: int = 512,
        # Actions
        action_vocab_size: int = 65536,
        action_embed_dim: int = 256,
        # Training
        token_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Validation
        if input_mode not in ["discrete", "continuous"]:
            raise ValueError(
                f"input_mode must be 'discrete' or 'continuous', got '{input_mode}'"
            )
        if not isinstance(T, int) or T <= 0:
            raise ValueError(f"T must be positive int, got {T}")
        if not isinstance(S, int) or S <= 0:
            raise ValueError(f"S must be positive int, got {S}")
        if not isinstance(vjepa_embed_dim, int) or vjepa_embed_dim <= 0:
            raise ValueError(
                f"vjepa_embed_dim must be positive int, got {vjepa_embed_dim}"
            )
        if not isinstance(num_factored_vocabs, int) or num_factored_vocabs <= 0:
            raise ValueError(
                f"num_factored_vocabs must be positive int, got {num_factored_vocabs}"
            )
        if not isinstance(factored_vocab_size, int) or factored_vocab_size <= 0:
            raise ValueError(
                f"factored_vocab_size must be positive int, got {factored_vocab_size}"
            )
        if not isinstance(token_loss_weight, (int, float)) or token_loss_weight < 0:
            raise ValueError(
                f"token_loss_weight must be non-negative number, got {token_loss_weight}"
            )

        # Model config
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone

        # Data dimensions
        self.T = T
        self.S = S
        self.image_vocab_size = image_vocab_size

        # Input mode
        self.input_mode = input_mode
        self.vjepa_embed_dim = vjepa_embed_dim

        # Factorization
        self.num_factored_vocabs = num_factored_vocabs
        self.factored_vocab_size = factored_vocab_size

        # Actions
        self.action_vocab_size = action_vocab_size
        self.action_embed_dim = action_embed_dim

        # Training
        self.token_loss_weight = token_loss_weight

    def shallow_copy(self):
        return VJEPAPredictorConfig(**self.to_dict())
