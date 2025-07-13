"""
V-JEPA2-AC based world model for 1X challenge.
Replaces GENIE with V-JEPA2-AC backbone + factorized token prediction heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .config import VJEPAConfig


class VJEPAModelOutput(ModelOutput):
    """Output of V-JEPA world model"""
    loss: Optional[torch.FloatTensor] = None
    token_loss: Optional[torch.FloatTensor] = None
    l1_loss: Optional[torch.FloatTensor] = None
    acc: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


class ActionEmbedding(nn.Module):
    """Embed scalar action tokens for V-JEPA2-AC predictor"""
    
    def __init__(self, action_vocab_size: int, embed_dim: int):
        super().__init__()
        self.action_embed = nn.Embedding(action_vocab_size, embed_dim)
        
    def forward(self, action_tokens: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            action_tokens: [B, T] scalar action tokens (uint16)
        Returns:
            action_embeds: [B, T, embed_dim] 
        """
        return self.action_embed(action_tokens)


class FactorizedTokenPredictor(nn.Module):
    """Factorized token prediction heads for 1X challenge"""
    
    def __init__(self, embed_dim: int, factored_vocab_size: int = 512):
        super().__init__()
        self.token_head_1 = nn.Linear(embed_dim, factored_vocab_size, bias=True)
        self.token_head_2 = nn.Linear(embed_dim, factored_vocab_size, bias=True)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, N, embed_dim] predictor outputs
        Returns:
            logits_1, logits_2: [B, N, 512] factorized predictions
        """
        return self.token_head_1(x), self.token_head_2(x)


class VJEPAWorldModel(PreTrainedModel):
    """
    V-JEPA2-AC based world model for 1X robotics challenge.
    
    Architecture:
    1. Load pretrained V-JEPA2-AC encoder + predictor
    2. Replace continuous predictor output with factorized token heads
    3. Add action embedding for scalar action tokens
    4. Optional: Add flow matching loss in representation space
    """
    
    config_class = VJEPAConfig
    
    def __init__(self, config: VJEPAConfig):
        super().__init__(config)
        self.config = config
        
        # Load pretrained V-JEPA2-AC backbone
        self._load_vjepa_backbone()
        
        # Action embedding for 1X scalar actions
        self.action_embedding = ActionEmbedding(
            config.action_vocab_size, 
            config.action_embed_dim
        )
        
        # Replace V-JEPA2-AC's continuous output with factorized token prediction
        self.token_predictor = FactorizedTokenPredictor(
            embed_dim=self.encoder.embed_dim,  # 1408 for ViT-Giant
            factored_vocab_size=config.factored_vocab_size
        )
        
        # Store loss type for forward pass
        self.loss_type = config.loss_type
        
        # Initialize new components
        self._init_new_weights()
        
        # Freeze backbone if requested
        if config.freeze_backbone:
            self._freeze_backbone()
        
    def _load_vjepa_backbone(self):
        """Load pretrained V-JEPA2-AC encoder and predictor"""
        try:
            # This will be the actual V-JEPA2 import when integrated
            # from vjepa2.src.hub.backbones import vjepa2_ac_vit_giant
            # self.encoder, self.predictor = vjepa2_ac_vit_giant(pretrained=self.config.pretrained)
            
            # Placeholder for now - will be replaced with actual V-JEPA2 loading
            print("Loading V-JEPA2-AC backbone...")
            print(f"Model: {self.config.model_name}")
            print(f"Pretrained: {self.config.pretrained}")
            
            # TODO: Implement actual V-JEPA2-AC loading
            # For now, create placeholder modules
            class PlaceholderEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed_dim = 1408  # ViT-Giant embed dim
                    
                def forward(self, x):
                    B, T, H, W = x.shape
                    return torch.randn(B, T * H * W, self.embed_dim)
            
            class PlaceholderPredictor(nn.Module):
                def __init__(self):
                    super().__init__()
                    
                def forward(self, x, actions, states, extrinsics=None):
                    return x  # Pass through for now
            
            self.encoder = PlaceholderEncoder()
            self.predictor = PlaceholderPredictor()
            
        except ImportError:
            raise ImportError(
                "V-JEPA2 not found. Please ensure V-JEPA2 is installed and available."
            )
    
    def _init_new_weights(self):
        """Initialize newly added components"""
        # Initialize action embedding
        nn.init.normal_(self.action_embedding.action_embed.weight, std=0.02)
        
        # Initialize token prediction heads
        nn.init.xavier_uniform_(self.token_predictor.token_head_1.weight)
        nn.init.xavier_uniform_(self.token_predictor.token_head_2.weight)
        nn.init.zeros_(self.token_predictor.token_head_1.bias)
        nn.init.zeros_(self.token_predictor.token_head_2.bias)
    
    def _freeze_backbone(self):
        """Freeze V-JEPA encoder for transfer learning"""
        print("Freezing V-JEPA backbone...")
        
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Freeze predictor parameters (keep only token prediction heads trainable)
        for param in self.predictor.parameters():
            param.requires_grad = False
            
        # Keep action embedding and token predictor trainable
        for param in self.action_embedding.parameters():
            param.requires_grad = True
            
        for param in self.token_predictor.parameters():
            param.requires_grad = True
            
        # Print trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"Frozen params: {total_params - trainable_params:,} ({(total_params - trainable_params)/1e6:.1f}M)")
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        action_tokens: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> VJEPAModelOutput:
        """
        Args:
            input_ids: [B, T*H*W] flattened image tokens 
            action_tokens: [B, T] scalar action tokens (optional)
            labels: [B, T*H*W] ground truth tokens for loss computation
        """
        B, seq_len = input_ids.shape
        T, H, W = self.config.T, int(self.config.S**0.5), int(self.config.S**0.5)
        
        # Reshape to spatial-temporal format
        x_THW = input_ids.view(B, T, H, W)
        
        # Encode video frames
        z = self.encoder(x_THW)  # [B, T*H*W, embed_dim]
        
        # Prepare actions if provided
        if action_tokens is not None:
            action_embeds = self.action_embedding(action_tokens)  # [B, T, action_embed_dim]
            states = action_embeds  # Use actions as states for V-JEPA2-AC
            
            # Run V-JEPA2-AC predictor with action conditioning
            z_pred = self.predictor(z, action_embeds, states)
        else:
            # No action conditioning
            z_pred = z
        
        # Predict factorized tokens
        logits_1, logits_2 = self.token_predictor(z_pred)
        
        # Compute losses if labels provided
        total_loss = None
        token_loss = None
        l1_loss = None
        acc = None
        
        if labels is not None:
            # Factorized cross-entropy loss (same as GENIE)
            from genie.factorization_utils import factorize_labels
            
            labels_THW = labels.view(B, T, H, W)
            factored_labels = factorize_labels(
                labels_THW, 
                self.config.num_factored_vocabs, 
                self.config.factored_vocab_size
            )
            
            # Combine factorized logits for loss computation
            factored_logits = torch.stack([logits_1, logits_2], dim=1)  # [B, 2, N, 512]
            factored_logits = factored_logits.permute(0, 3, 1, 2)  # [B, 512, 2, N]
            
            # Cross-entropy loss on factorized vocabulary
            token_loss = F.cross_entropy(
                factored_logits.flatten(0, 1), 
                factored_labels.flatten(0, 1), 
                reduction="none"
            ).view(B, 2, -1).sum(dim=1).mean()
            
            # Accuracy
            pred_tokens_1 = logits_1.argmax(dim=-1)
            pred_tokens_2 = logits_2.argmax(dim=-1)
            acc = ((pred_tokens_1 == factored_labels[:, 0]) & 
                    (pred_tokens_2 == factored_labels[:, 1])).float().mean()
                
            total_loss = self.config.token_loss_weight * token_loss
        
        return VJEPAModelOutput(
            loss=total_loss,
            token_loss=token_loss,
            l1_loss=l1_loss,
            acc=acc,
            logits=torch.stack([logits_1, logits_2], dim=1)  # [B, 2, N, 512]
        )
    
    def generate(
        self, 
        input_ids: torch.LongTensor,
        action_tokens: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 256,
        **kwargs
    ):
        """Generate future tokens autoregressively"""
        # TODO: Implement generation logic similar to GENIE
        # This would use the factorized token heads to predict next frame tokens
        pass


def create_vjepa_model(config_path: Optional[str] = None, **kwargs) -> VJEPAWorldModel:
    """Factory function to create V-JEPA world model"""
    if config_path:
        config = VJEPAConfig.from_pretrained(config_path)
    else:
        config = VJEPAConfig(**kwargs)
    
    return VJEPAWorldModel(config)