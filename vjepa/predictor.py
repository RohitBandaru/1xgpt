"""
V-JEPA Predictor for 1X Challenge

Loads V-JEPA2-AC predictor and trains a factorized output layer on top.
Works with both V-JEPA tokens and COSMOS tokens.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class FactorizedTokenPredictor(nn.Module):
    """Factorized token prediction heads for 1X challenge"""

    def __init__(self, embed_dim: int, factored_vocab_size: int = 512):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Linear(embed_dim, factored_vocab_size, bias=True),
                nn.Linear(embed_dim, factored_vocab_size, bias=True),
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.heads[0](x), self.heads[1](x)


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

        # Input processing components
        self.input_mode = getattr(config, "input_mode", "discrete")
        self.token_embedding = nn.Embedding(
            config.image_vocab_size + 1, config.vjepa_embed_dim
        )
        self.continuous_proj = (
            nn.Linear(config.vjepa_embed_dim, config.vjepa_embed_dim)
            if self.input_mode in ["continuous", "hybrid"]
            else None
        )

        # Action processing
        self.action_embedding = nn.Embedding(
            config.action_vocab_size, config.action_embed_dim
        )
        self.action_to_state = nn.Linear(config.action_embed_dim, 7)

        # Factorized output head for 1X challenge
        # V-JEPA AC predictor projects back to encoder dimension (embed_dim)
        self.factorized_head = FactorizedTokenPredictor(
            embed_dim=config.vjepa_embed_dim,  # V-JEPA predictor outputs embed_dim (1408 for ViT-G)
            factored_vocab_size=config.factored_vocab_size,
        )

        # Initialize and optionally freeze
        self._init_weights()
        if config.freeze_backbone:
            self._freeze_predictor()

    def _load_vjepa_predictor(self):
        """Load V-JEPA2-AC predictor from PyTorch Hub"""
        print("Loading V-JEPA2-AC predictor...")

        try:
            if self.config.pretrained:
                # Load pretrained V-JEPA predictor (standard dimensions)
                _, predictor = torch.hub.load(
                    "facebookresearch/vjepa2",
                    "vjepa2_ac_vit_giant",
                    pretrained=True,
                    verbose=False,
                )
                self.predictor = predictor

                print(f"✅ Loaded pretrained V-JEPA predictor:")
                print(
                    f"   Parameters: {sum(p.numel() for p in self.predictor.parameters()):,}"
                )
            else:
                # Load architecture without pretrained weights (standard dimensions)
                _, predictor = torch.hub.load(
                    "facebookresearch/vjepa2",
                    "vjepa2_ac_vit_giant",
                    pretrained=False,
                    verbose=False,
                )
                self.predictor = predictor

                print(
                    f"✅ Loaded V-JEPA predictor architecture (no pretrained weights):"
                )
                print(
                    f"   Parameters: {sum(p.numel() for p in self.predictor.parameters()):,}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load V-JEPA predictor from PyTorch Hub: {e}")

        # Validate loaded predictor has expected attributes
        required_attrs = ["grid_height", "grid_width", "num_frames", "tubelet_size"]
        for attr in required_attrs:
            if not hasattr(self.predictor, attr):
                raise AttributeError(
                    f"Loaded V-JEPA predictor missing required attribute: {attr}"
                )

        # Store V-JEPA's expected dimensions for input processing
        self.vjepa_spatial_patches = (
            self.predictor.grid_height * self.predictor.grid_width
        )  # 256
        self.vjepa_temporal_steps = (
            self.predictor.num_frames // self.predictor.tubelet_size
        )  # 32
        self.vjepa_total_patches = (
            self.vjepa_temporal_steps * self.vjepa_spatial_patches
        )  # 8192

        print(
            f"   V-JEPA expects: {self.vjepa_spatial_patches} spatial patches, {self.vjepa_temporal_steps} temporal steps"
        )
        print(f"   Total patches expected: {self.vjepa_total_patches}")
        print(
            f"   Config provides: {self.config.S} spatial, {self.config.T} temporal = {self.config.T * self.config.S} total"
        )

    def _init_weights(self):
        """Initialize new components with standard initialization"""
        for module in [
            self.token_embedding,
            self.action_embedding,
            self.action_to_state,
            self.factorized_head,
        ]:
            if hasattr(module, "weight"):
                if len(module.weight.shape) > 1:
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.normal_(module.weight, std=0.02)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

        if self.continuous_proj is not None:
            nn.init.xavier_uniform_(self.continuous_proj.weight)
            nn.init.zeros_(self.continuous_proj.bias)

    def _freeze_predictor(self):
        """Freeze V-JEPA predictor for transfer learning"""
        for param in self.predictor.parameters():
            param.requires_grad = False

    def _process_inputs(self, input_ids=None, input_embeds=None):
        """Process inputs and resize to V-JEPA format"""
        if input_ids is not None and input_embeds is not None:
            raise ValueError("Cannot specify both input_ids and input_embeds")
        if input_ids is None and input_embeds is None:
            raise ValueError("Must specify either input_ids or input_embeds")

        if input_ids is not None:
            B, N = input_ids.shape
            token_embeds = self.token_embedding(input_ids)
        else:
            if self.continuous_proj is None:
                raise ValueError("Model not configured for continuous inputs")
            B, N, _ = input_embeds.shape
            token_embeds = self.continuous_proj(input_embeds)

        return self._resize_to_vjepa_format(token_embeds, B, N)

    def _resize_to_vjepa_format(self, token_embeds, B, N):
        """Resize embeddings to match V-JEPA expected dimensions"""
        if N == self.vjepa_total_patches:
            return token_embeds

        if N < self.vjepa_total_patches:
            pad_size = self.vjepa_total_patches - N
            padding = torch.zeros(
                B,
                pad_size,
                token_embeds.size(-1),
                device=token_embeds.device,
                dtype=token_embeds.dtype,
            )
            return torch.cat([token_embeds, padding], dim=1)
        else:
            return token_embeds[:, : self.vjepa_total_patches]

    def _process_actions(self, action_tokens, B, T):
        """Process action tokens into robot states"""
        if action_tokens is None:
            device = next(self.parameters()).device
            return torch.zeros(B, T, 7, device=device)

        action_T = action_tokens.size(1)
        if action_T != T:
            if action_T < T:
                last_actions = action_tokens[:, -1:].repeat(1, T - action_T)
                action_tokens = torch.cat([action_tokens, last_actions], dim=1)
            else:
                action_tokens = action_tokens[:, :T]

        action_embeds = self.action_embedding(action_tokens)
        return self.action_to_state(action_embeds)

    def _align_labels(self, labels, N, B):
        """Align labels with resized inputs"""
        if labels is None:
            return None

        original_N = labels.size(1)
        if original_N == N:
            return labels

        if original_N < N:
            pad_size = N - original_N
            padding = torch.zeros(B, pad_size, dtype=labels.dtype, device=labels.device)
            return torch.cat([labels, padding], dim=1)
        else:
            return labels[:, :N]

    def _compute_loss(self, logits_1, logits_2, labels, B, T, N):
        """Compute factorized loss"""
        from genie.factorization_utils import factorize_labels

        # Reshape and factorize
        N_spatial = N // T
        H_tokens = W_tokens = int(N_spatial**0.5)

        labels_THW = labels.view(B, T, H_tokens, W_tokens)
        factored_labels = factorize_labels(
            labels_THW, self.config.num_factored_vocabs, self.config.factored_vocab_size
        )

        # Compute losses
        logits_1_THW = logits_1.view(B, T, H_tokens, W_tokens, -1)
        logits_2_THW = logits_2.view(B, T, H_tokens, W_tokens, -1)

        loss_1 = F.cross_entropy(
            logits_1_THW.reshape(-1, self.config.factored_vocab_size),
            factored_labels[:, 0].reshape(-1),
            reduction="mean",
        )
        loss_2 = F.cross_entropy(
            logits_2_THW.reshape(-1, self.config.factored_vocab_size),
            factored_labels[:, 1].reshape(-1),
            reduction="mean",
        )

        token_loss = loss_1 + loss_2
        total_loss = self.config.token_loss_weight * token_loss

        # Compute accuracy
        pred_1 = logits_1_THW.argmax(dim=-1)
        pred_2 = logits_2_THW.argmax(dim=-1)
        acc = (
            ((pred_1 == factored_labels[:, 0]) & (pred_2 == factored_labels[:, 1]))
            .float()
            .mean()
        )

        return total_loss, token_loss, acc

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_embeds: Optional[torch.FloatTensor] = None,
        action_tokens: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> VJEPAModelOutput:
        """Simplified forward pass for V-JEPA predictor"""

        # 1. Process inputs (unified for discrete/continuous)
        token_embeds = self._process_inputs(input_ids, input_embeds)
        B, N, _ = token_embeds.shape
        T = self.vjepa_temporal_steps

        # 2. Process actions into robot states
        robot_states = self._process_actions(action_tokens, B, T)

        # 3. V-JEPA forward pass (using same states for actions and poses)
        z_pred = self.predictor(token_embeds, robot_states, robot_states)

        # 4. Generate factorized predictions
        logits_1, logits_2 = self.factorized_head(z_pred)

        # 5. Compute loss if labels provided
        total_loss, token_loss, acc = None, None, None
        if labels is not None:
            labels = self._align_labels(labels, N, B)
            total_loss, token_loss, acc = self._compute_loss(
                logits_1, logits_2, labels, B, T, N
            )

        # Return compatible output format
        if labels is not None:
            # GENIE-compatible format for training
            H = W = int((N // T) ** 0.5)
            logits_1_CTHW = logits_1.view(B, T, H, W, -1).permute(0, 4, 1, 2, 3)
            logits_2_CTHW = logits_2.view(B, T, H, W, -1).permute(0, 4, 1, 2, 3)
            combined_logits = torch.cat([logits_1_CTHW, logits_2_CTHW], dim=1)
            return ModelOutput(loss=total_loss, acc=acc, logits=combined_logits)
        else:
            # Original format for generation
            return VJEPAModelOutput(
                loss=total_loss,
                token_loss=token_loss,
                acc=acc,
                logits=torch.stack([logits_1, logits_2], dim=1),
            )

    def generate(
        self,
        input_ids: torch.LongTensor,
        action_tokens: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs,
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
                    input_ids=all_tokens, action_tokens=action_tokens
                )

                # Get logits for next token
                logits_1, logits_2 = (
                    outputs.logits[:, 0, -1:],
                    outputs.logits[:, 1, -1:],
                )  # [B, 1, 512]

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
                next_token = (
                    next_token_1 * self.config.factored_vocab_size + next_token_2
                )

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
            raise ValueError(
                "Cannot set continuous mode: model not configured for V-JEPA embeddings"
            )

        self.input_mode = mode
        print(f"✅ Input mode set to: {mode}")
