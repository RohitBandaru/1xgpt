"""
V-JEPA Predictor for 1X Challenge

Loads V-JEPA2-AC predictor and trains a factorized output layer on top.
Works with both V-JEPA tokens and COSMOS tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .config import VJEPAPredictorConfig


@dataclass
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


class VJEPAPredictor(PreTrainedModel):
    """
    V-JEPA Predictor model for the 1X challenge.
    
    Loads pretrained V-JEPA2-AC predictor and adds factorized output heads.
    Supports both V-JEPA tokens (~8K vocab) and COSMOS tokens (262K vocab).
    """
    
    config_class = VJEPAPredictorConfig
    
    def __init__(self, config: VJEPAPredictorConfig):
        super().__init__(config)
        self.config = config
        
        # Load V-JEPA2-AC predictor
        self._load_vjepa_predictor()
        
        # Action embedding for 1X scalar actions
        self.action_embedding = ActionEmbedding(
            config.action_vocab_size,
            config.action_embed_dim
        )
        # V-JEPA expects 7-dim robot states, not high-dim embeddings
        self.action_to_state = nn.Linear(config.action_embed_dim, 7)  # Convert to 7D robot state
        
        # Dual input support: COSMOS tokens OR continuous V-JEPA embeddings
        self.input_mode = getattr(config, 'input_mode', 'discrete')  # 'discrete' or 'continuous'
        
        # Always support COSMOS tokens (from 1X precomputed dataset)
        # V-JEPA predictor expects 1408-dim embeddings (ViT-G dimension)
        # Add +1 to vocab size to handle mask token (image_vocab_size)
        self.token_embedding = nn.Embedding(
            config.image_vocab_size + 1,  # COSMOS vocabulary size + mask token
            config.vjepa_embed_dim  # Use V-JEPA embed dim (1408) to match predictor
        )
        
        # Optionally support continuous V-JEPA embeddings
        if self.input_mode in ['continuous', 'hybrid']:
            vjepa_embed_dim = getattr(config, 'vjepa_embed_dim', 1408)  # ViT-G embedding dimension
            self.continuous_proj = nn.Linear(vjepa_embed_dim, config.vjepa_embed_dim)
        else:
            self.continuous_proj = None
        
        # Factorized output head for 1X challenge
        # V-JEPA predictor outputs its internal dimension, add projection if needed
        self.factorized_head = FactorizedTokenPredictor(
            embed_dim=config.pred_embed_dim,  # This will be the V-JEPA predictor's output dim
            factored_vocab_size=config.factored_vocab_size
        )
        
        # Initialize new components
        self._init_new_weights()
        
        # Freeze predictor if requested
        if config.freeze_backbone:
            self._freeze_predictor()
    
    def _load_vjepa_predictor(self):
        """Load V-JEPA2-AC predictor from PyTorch Hub"""
        print("Loading V-JEPA2-AC predictor...")
        
        if self.config.pretrained:
            # Load pretrained V-JEPA predictor
            _, predictor = torch.hub.load(
                'facebookresearch/vjepa2',
                'vjepa2_ac_vit_giant',
                pretrained=True,
                verbose=False
            )
            self.predictor = predictor
            
            print(f"✅ Loaded pretrained V-JEPA predictor:")
            print(f"   Parameters: {sum(p.numel() for p in self.predictor.parameters()):,}")
        else:
            # Load architecture without pretrained weights
            _, predictor = torch.hub.load(
                'facebookresearch/vjepa2',
                'vjepa2_ac_vit_giant',
                pretrained=False,
                verbose=False
            )
            self.predictor = predictor
            
            print(f"✅ Loaded V-JEPA predictor architecture (no pretrained weights):")
            print(f"   Parameters: {sum(p.numel() for p in self.predictor.parameters()):,}")
    
    def _init_new_weights(self):
        """Initialize newly added components"""
        # Initialize action embedding
        nn.init.normal_(self.action_embedding.action_embed.weight, std=0.02)
        
        # Initialize action to state projection
        nn.init.xavier_uniform_(self.action_to_state.weight)
        if self.action_to_state.bias is not None:
            nn.init.zeros_(self.action_to_state.bias)
        
        # Initialize COSMOS token embedding
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # Initialize continuous projection (if present)
        if self.continuous_proj is not None:
            nn.init.xavier_uniform_(self.continuous_proj.weight)
            if self.continuous_proj.bias is not None:
                nn.init.zeros_(self.continuous_proj.bias)
        
        # Initialize factorized head
        for module in self.factorized_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _freeze_predictor(self):
        """Freeze V-JEPA predictor for transfer learning"""
        print("Freezing V-JEPA predictor...")
        
        # Freeze predictor parameters
        for param in self.predictor.parameters():
            param.requires_grad = False
        
        # Keep new components trainable
        for param in self.action_embedding.parameters():
            param.requires_grad = True
        
        # Keep action projection trainable
        for param in self.action_to_state.parameters():
            param.requires_grad = True
        
        # Keep COSMOS token embedding trainable
        for param in self.token_embedding.parameters():
            param.requires_grad = True
        
        # Keep continuous projection trainable (if present)
        if self.continuous_proj is not None:
            for param in self.continuous_proj.parameters():
                param.requires_grad = True
        
        for param in self.factorized_head.parameters():
            param.requires_grad = True
        
        # Print parameter summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Parameter summary:")
        print(f"  Total: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"  Frozen: {frozen_params:,} ({frozen_params/1e6:.1f}M)")
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        action_tokens: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> VJEPAModelOutput:
        """
        Forward pass for V-JEPA predictor supporting COSMOS tokens and continuous V-JEPA embeddings.
        
        Args:
            input_ids: [B, T*N_patches] discrete COSMOS token sequence (from 1X dataset)
            input_embeds: [B, N_patches, embed_dim] continuous V-JEPA embeddings
            action_tokens: [B, T] scalar action tokens
            labels: [B, T*N_patches] ground truth COSMOS tokens for loss
            
        Returns:
            VJEPAModelOutput with loss, logits, and accuracy
        """
        # Validate inputs
        if input_ids is not None and input_embeds is not None:
            raise ValueError("Cannot specify both input_ids and input_embeds")
        if input_ids is None and input_embeds is None:
            raise ValueError("Must specify either input_ids or input_embeds")
        
        # 1. Process inputs based on type
        if input_ids is not None:
            # Discrete COSMOS token input
            B, N = input_ids.shape  # N = T * H_tokens * W_tokens
            token_embeds = self.token_embedding(input_ids)  # [B, N, embed_dim]
        else:
            # Continuous V-JEPA embedding input
            if self.continuous_proj is None:
                raise ValueError("Model not configured for continuous inputs. Use input_mode='continuous' or 'hybrid'")
            B, N, embed_dim = input_embeds.shape
            token_embeds = self.continuous_proj(input_embeds)  # [B, N, predictor_embed_dim]
        
        # 2. Prepare action conditioning - V-JEPA predictor expects 7D robot states
        if action_tokens is not None:
            action_embeds = self.action_embedding(action_tokens)  # [B, T, action_embed_dim]
            robot_states = self.action_to_state(action_embeds)  # [B, T, 7] - 7D robot states
            
            # V-JEPA expects states at each timestep, not expanded to spatial locations
            # Use robot states directly for both actions and poses
            action_states = robot_states  # [B, T, 7]
            pose_states = robot_states    # [B, T, 7] - same as actions for simplicity
        else:
            # No action conditioning - create dummy 7D robot states
            T = self.config.T  # Use config T for temporal dimension
            device = token_embeds.device
            
            # Create zero robot states with correct 7D shape
            action_states = torch.zeros(B, T, 7, device=device)
            pose_states = torch.zeros(B, T, 7, device=device)
        
        # Always call V-JEPA predictor with 3 arguments: video_representations, actions, states
        # V-JEPA2-AC expects: (video_tokens, action_deltas, robot_states)
        z_pred = self.predictor(token_embeds, action_states, pose_states)
        
        # 3. Convert to factorized tokens for 1X challenge
        logits_1, logits_2 = self.factorized_head(z_pred)  # [B, N, 512] each
        
        # 4. Compute loss if labels provided
        total_loss = None
        token_loss = None
        acc = None
        
        if labels is not None:
            from genie.factorization_utils import factorize_labels
            
            # Determine spatial dimensions from sequence length
            T = action_tokens.size(1) if action_tokens is not None else 16  # Default
            N_spatial = N // T
            H_tokens = W_tokens = int(N_spatial**0.5)
            
            # Factorize labels to match prediction format
            labels_THW = labels.view(B, T, H_tokens, W_tokens)
            factored_labels = factorize_labels(
                labels_THW,
                self.config.num_factored_vocabs,
                self.config.factored_vocab_size
            )  # [B, 2, T, H, W]
            
            # Reshape predictions to spatial-temporal format
            logits_1_THW = logits_1.view(B, T, H_tokens, W_tokens, -1)
            logits_2_THW = logits_2.view(B, T, H_tokens, W_tokens, -1)
            
            # Compute factorized cross-entropy loss
            loss_1 = F.cross_entropy(
                logits_1_THW.reshape(-1, self.config.factored_vocab_size),
                factored_labels[:, 0].reshape(-1),
                reduction="mean"
            )
            
            loss_2 = F.cross_entropy(
                logits_2_THW.reshape(-1, self.config.factored_vocab_size),
                factored_labels[:, 1].reshape(-1),
                reduction="mean"
            )
            
            token_loss = loss_1 + loss_2
            total_loss = self.config.token_loss_weight * token_loss
            
            # Compute accuracy
            pred_1 = logits_1_THW.argmax(dim=-1)
            pred_2 = logits_2_THW.argmax(dim=-1)
            acc_mask = (pred_1 == factored_labels[:, 0]) & (pred_2 == factored_labels[:, 1])
            acc = acc_mask.float().mean()
        
        # For compatibility with GENIE evaluation framework, format as GENIE-style output
        if labels is not None:
            # Format logits to match GENIE: [B, C, T, H, W] where C = factored_vocab_size * num_factored_vocabs  
            B, N = logits_1.shape[:2]
            T = self.config.T
            H = W = int((N // T) ** 0.5)
            
            # Reshape and combine factorized logits
            logits_1_THW = logits_1.view(B, T, H, W, -1).permute(0, 4, 1, 2, 3)  # [B, 512, T, H, W]
            logits_2_THW = logits_2.view(B, T, H, W, -1).permute(0, 4, 1, 2, 3)  # [B, 512, T, H, W]
            combined_logits = torch.cat([logits_1_THW, logits_2_THW], dim=1)  # [B, 1024, T, H, W]
            
            return ModelOutput(loss=total_loss, acc=acc, logits=combined_logits)
        else:
            # Return original format for generation/inference
            return VJEPAModelOutput(
                loss=total_loss,
                token_loss=token_loss,
                l1_loss=None,
                acc=acc,
                logits=torch.stack([logits_1, logits_2], dim=1)  # [B, 2, N, 512]
            )
    
    def generate(
        self,
        input_ids: torch.LongTensor,
        action_tokens: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate future tokens using V-JEPA predictor.
        
        Args:
            input_ids: [B, N_context] context tokens
            action_tokens: [B, T_new] action tokens for generation
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            generated_tokens: [B, N_context + max_new_tokens] all tokens
        """
        B, N_context = input_ids.shape
        all_tokens = input_ids.clone()
        
        # Generate tokens autoregressively
        for step in range(max_new_tokens):
            with torch.no_grad():
                # Forward pass on current sequence
                outputs = self.forward(
                    input_ids=all_tokens,
                    action_tokens=action_tokens
                )
                
                # Get logits for next token
                logits_1, logits_2 = outputs.logits[:, 0, -1:], outputs.logits[:, 1, -1:]  # [B, 1, 512]
                
                # Sample next token
                if temperature <= 1e-8:
                    # Greedy sampling
                    next_token_1 = logits_1.argmax(dim=-1)
                    next_token_2 = logits_2.argmax(dim=-1)
                else:
                    # Temperature sampling
                    probs_1 = F.softmax(logits_1 / temperature, dim=-1)
                    probs_2 = F.softmax(logits_2 / temperature, dim=-1)
                    next_token_1 = torch.multinomial(probs_1.squeeze(1), 1)
                    next_token_2 = torch.multinomial(probs_2.squeeze(1), 1)
                
                # Combine factorized tokens
                next_token = next_token_1 * self.config.factored_vocab_size + next_token_2
                
                # Append to sequence
                all_tokens = torch.cat([all_tokens, next_token], dim=1)
        
        return all_tokens
    
    def set_input_mode(self, mode: str):
        """
        Switch between discrete COSMOS tokens and continuous V-JEPA embeddings.
        
        Args:
            mode: "discrete" for COSMOS tokens, "continuous" for V-JEPA embeddings
        """
        if mode not in ["discrete", "continuous"]:
            raise ValueError(f"mode must be 'discrete' or 'continuous', got {mode}")
        
        # Check if the requested mode is supported by current configuration
        if mode == "continuous" and self.continuous_proj is None:
            raise ValueError("Cannot set continuous mode: model not configured for V-JEPA embeddings")
        
        self.input_mode = mode
        print(f"✅ Input mode set to: {mode}")
    
