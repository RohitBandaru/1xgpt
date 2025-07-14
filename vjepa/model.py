"""
V-JEPA2-AC based world model for 1X challenge.
Replaces GENIE with V-JEPA2-AC backbone + factorized token prediction heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .config import VJEPAConfig


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
        
        # Token embedding layer to convert discrete tokens to continuous embeddings
        # This bridges the gap between tokenized input and V-JEPA's continuous representations
        # +1 to account for mask token at image_vocab_size
        self.token_embedding = nn.Embedding(config.image_vocab_size + 1, self.encoder.embed_dim)
        
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
        if self.config.pretrained:
            # Load V-JEPA2-AC via PyTorch Hub (as per README instructions)
            print("Loading V-JEPA2-AC backbone via PyTorch Hub...")
            print(f"Pretrained: {self.config.pretrained}")
            
            # Load V-JEPA2-AC without conflicting parameters
            self.encoder, self.predictor = torch.hub.load(
                'facebookresearch/vjepa2', 
                'vjepa2_ac_vit_giant', 
                pretrained=self.config.pretrained,
                verbose=False
            )
            
            print(f"Loaded V-JEPA2-AC ViT-Giant:")
            print(f"  Encoder embed_dim: {self.encoder.embed_dim}")
            print(f"  Encoder depth: {getattr(self.encoder, 'depth', 'N/A')}")
            print(f"  Encoder num_heads: {getattr(self.encoder, 'num_heads', 'N/A')}")
        else:
            # Create placeholder encoder and predictor for testing/development
            print("Creating placeholder V-JEPA backbone (pretrained=False)")
            
            class PlaceholderEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed_dim = 1408  # ViT-Giant embed_dim
                    self.depth = 40
                    self.num_heads = 16
                    # Simple placeholder that just processes tokens
                    self.linear = nn.Linear(self.embed_dim, self.embed_dim)
                    
                def forward(self, x):
                    return self.linear(x)
            
            class PlaceholderPredictor(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed_dim = 1408
                    self.linear = nn.Linear(self.embed_dim, self.embed_dim)
                    
                def forward(self, x, action_embeds=None, states=None):
                    return self.linear(x)
            
            self.encoder = PlaceholderEncoder()
            self.predictor = PlaceholderPredictor()
            
            print(f"Created placeholder V-JEPA backbone:")
            print(f"  Encoder embed_dim: {self.encoder.embed_dim}")
            print(f"  Encoder depth: {self.encoder.depth}")
            print(f"  Encoder num_heads: {self.encoder.num_heads}")
    
    def _init_new_weights(self):
        """Initialize newly added components"""
        # Initialize token embedding
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
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
            
        # Keep token embedding, action embedding and token predictor trainable
        for param in self.token_embedding.parameters():
            param.requires_grad = True
            
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
        
        # Identify masked positions (same as GENIE approach)
        mask_token_id = self.config.image_vocab_size
        # GENIE excludes first frame from loss computation: x_THW[:, 1:]
        relevant_mask = x_THW[:, 1:] == mask_token_id  # [B, T-1, H, W] 
        masked_positions = torch.zeros_like(x_THW, dtype=torch.bool)
        masked_positions[:, 1:] = relevant_mask  # Only frames 1 to T-1
        
        # Convert discrete tokens to continuous embeddings
        # Handle mask tokens properly - they should get special mask embeddings
        x_emb = self.token_embedding(x_THW.long())  # [B, T, H, W, embed_dim]
        
        # Flatten to sequence format expected by V-JEPA transformer blocks
        z = x_emb.view(B, T * H * W, self.encoder.embed_dim)  # [B, T*H*W, embed_dim]
        
        # Store masked positions for loss computation (flatten to match z)
        masked_positions_flat = masked_positions.view(B, T * H * W)  # [B, T*H*W]
        
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
            
            # ALIGNED WITH GENIE: Compute loss exactly like GENIE does
            # Only on masked positions, excluding first frame, using GENIE's reduction strategy
            
            # Reshape labels to match GENIE format: exclude first frame
            labels_THW_no_first = labels_THW[:, 1:]  # [B, T-1, H, W] - like GENIE
            factored_labels_no_first = factorize_labels(
                labels_THW_no_first,
                self.config.num_factored_vocabs,
                self.config.factored_vocab_size
            )  # [B, 2, T-1, H, W]
            
            # Reshape predictions to match (exclude first frame predictions)
            # logits_1/2 are [B, T*H*W, 512], reshape and exclude first frame
            logits_1_THW = logits_1.view(B, T, H, W, -1)[:, 1:]  # [B, T-1, H, W, 512]
            logits_2_THW = logits_2.view(B, T, H, W, -1)[:, 1:]  # [B, T-1, H, W, 512]
            
            # SIMPLIFIED: Follow GENIE's approach exactly but with simpler tensor manipulation
            # Process each factorized vocabulary separately (like before), then sum
            loss_1 = F.cross_entropy(
                logits_1_THW.reshape(-1, 512),  # [B*(T-1)*H*W, 512] 
                factored_labels_no_first[:, 0].reshape(-1),  # [B*(T-1)*H*W]
                reduction="none"
            ).view(B, T-1, H, W)  # [B, T-1, H, W]
            
            loss_2 = F.cross_entropy(
                logits_2_THW.reshape(-1, 512),  # [B*(T-1)*H*W, 512]
                factored_labels_no_first[:, 1].reshape(-1),  # [B*(T-1)*H*W] 
                reduction="none"
            ).view(B, T-1, H, W)  # [B, T-1, H, W]
            
            # Sum losses across factorized vocabs (equivalent to GENIE's .sum(dim=1))
            loss_THW = loss_1 + loss_2  # [B, T-1, H, W]
            
            # Compute mean loss over masked positions only (like GENIE)
            relevant_mask_THW = relevant_mask  # [B, T-1, H, W]
            num_masked_tokens = torch.sum(relevant_mask_THW)
            if num_masked_tokens > 0:
                token_loss = torch.sum(loss_THW * relevant_mask_THW) / num_masked_tokens
            else:
                token_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            
            # Accuracy computation exactly like GENIE: require ALL factorized vocabs to be correct
            pred_1 = logits_1_THW.argmax(dim=-1)  # [B, T-1, H, W] - predictions for vocab 1
            pred_2 = logits_2_THW.argmax(dim=-1)  # [B, T-1, H, W] - predictions for vocab 2
            
            # GENIE's .all(dim=1) logic: ALL factorized vocabs must be correct
            acc_THW = ((pred_1 == factored_labels_no_first[:, 0]) & 
                       (pred_2 == factored_labels_no_first[:, 1]))  # [B, T-1, H, W]
            
            # Mean accuracy over masked positions only (like GENIE)
            if num_masked_tokens > 0:
                acc = torch.sum(acc_THW * relevant_mask_THW).float() / num_masked_tokens
            else:
                acc = torch.tensor(0.0, device=input_ids.device)
                
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
        attention_mask: torch.LongTensor = None,
        action_tokens: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 256,
        min_new_tokens: int = None,
        temperature: float = 0.0,
        return_logits: bool = False,
        **kwargs
    ) -> torch.LongTensor:
        """Generate future tokens frame-by-frame using V-JEPA predictions
        
        Follows GENIE's generation interface for compatibility with evaluation pipeline.
        
        Args:
            input_ids: [B, T*H*W] flattened image tokens for context frames
            attention_mask: Ignored (for compatibility with GENIE interface)
            action_tokens: [B, T_new] scalar action tokens for generation steps
            max_new_tokens: Number of new tokens to generate (must be multiple of S)
            min_new_tokens: If specified, must equal max_new_tokens (for compatibility)
            temperature: Sampling temperature (0.0 for greedy)
            return_logits: Whether to return logits (for compatibility)
            
        Returns:
            generated_tokens: [B, (T + T_new)*H*W] input + generated tokens
        """
        assert min_new_tokens in (None, max_new_tokens), \
            "Expecting `min_new_tokens`, if specified, to match `max_new_tokens`."
        
        assert max_new_tokens % self.config.S == 0, "max_new_tokens must be multiple of S"
        num_new_frames = max_new_tokens // self.config.S
        
        B, seq_len = input_ids.shape
        H, W = int(self.config.S**0.5), int(self.config.S**0.5)
        
        # Reshape input to spatial-temporal format
        inputs_THW = input_ids.view(B, -1, H, W)  # [B, T_prompt, H, W]
        
        # Create full sequence with masked future frames (like GENIE)
        mask_token_id = self.config.image_vocab_size
        masked_future = torch.full(
            (B, num_new_frames, H, W),
            mask_token_id, 
            dtype=torch.long, 
            device=input_ids.device
        )
        full_sequence = torch.cat([inputs_THW, masked_future], dim=1)  # [B, T_total, H, W]
        
        # Generate frames one by one
        all_logits = []
        for frame_idx in range(num_new_frames):
            timestep = inputs_THW.size(1) + frame_idx
            
            with torch.no_grad():
                # Forward pass on full sequence with masked future
                sequence_flat = full_sequence.view(B, -1)
                outputs = self.forward(input_ids=sequence_flat, action_tokens=action_tokens)
                
                # Extract logits for current frame
                logits = outputs.logits  # [B, 2, T*H*W, 512]
                
                # Get logits for the frame we're generating
                frame_start = timestep * H * W
                frame_end = frame_start + H * W
                frame_logits = logits[:, :, frame_start:frame_end, :]  # [B, 2, H*W, 512]
                
                # Sample from factorized distributions (follow GENIE's approach exactly)
                next_frame_tokens = torch.zeros(B, H, W, dtype=torch.long, device=input_ids.device)
                
                # Process vocabularies in REVERSE order like GENIE (flip the factor indices)
                # GENIE uses .flip(2).unbind(2) which processes from highest to lowest order
                for factor_idx in reversed(range(self.config.num_factored_vocabs)):
                    factor_logits = frame_logits[:, factor_idx]  # [B, H*W, 512]
                    
                    if temperature <= 1e-8:
                        # Greedy sampling
                        factor_tokens = factor_logits.argmax(dim=-1)  # [B, H*W]
                    else:
                        # Temperature sampling
                        probs = F.softmax(factor_logits / temperature, dim=-1)
                        dist = torch.distributions.categorical.Categorical(probs=probs)
                        factor_tokens = dist.sample()  # [B, H*W]
                    
                    # Reshape to spatial and combine factors exactly like GENIE
                    factor_tokens_hw = factor_tokens.view(B, H, W)
                    
                    # GENIE's combination: samples_HW *= vocab_size; samples_HW += sample
                    next_frame_tokens *= self.config.factored_vocab_size
                    next_frame_tokens += factor_tokens_hw
                
                # Update the sequence with generated frame
                full_sequence[:, timestep] = next_frame_tokens
                
                if return_logits:
                    # Store logits in GENIE-compatible format
                    # frame_logits is [B, 2, H*W, 512], need to rearrange to [B, 512, 2, H, W]
                    frame_logits_reshaped = frame_logits.permute(0, 3, 1, 2).view(
                        B, self.config.factored_vocab_size, self.config.num_factored_vocabs, H, W
                    )
                    all_logits.append(frame_logits_reshaped)
        
        # Return in GENIE-compatible format
        predicted_tokens = full_sequence.view(B, -1)
        
        if return_logits:
            stacked_logits = torch.stack(all_logits, dim=3)  # [B, vocab_size, num_vocabs, num_frames, H, W]
            return predicted_tokens, stacked_logits
        else:
            return predicted_tokens

