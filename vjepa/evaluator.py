"""
V-JEPA-specific evaluator implementation.

This file contains the core evaluation logic extracted from the original
vjepa/evaluate.py file. The original file has been removed and its functionality
has been refactored into this model-specific evaluator class and the shared
evaluation framework in ../evaluate.py.

Original file: vjepa/evaluate.py
Refactored: 2025-07-14
"""

import torch
from typing import Tuple

from evaluate import BaseEvaluator
from vjepa.predictor import VJEPAPredictor


class VJEPAEvaluator(BaseEvaluator):
    """V-JEPA model evaluator."""
    
    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        super().__init__(checkpoint_dir, device)
        
    def _load_model(self):
        """Load V-JEPA model from checkpoint."""
        self.model = VJEPAPredictor.from_pretrained(self.checkpoint_dir)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict_and_get_logits(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Frame-by-frame prediction like GENIE's teacher-forced evaluation.
        
        For each timestep, predict the next frame using all previous frames as context.
        This matches GENIE's evaluation protocol for proper parity.
        
        Args:
            input_ids: [B, T*H*W] input tokens
            
        Returns:
            predicted_tokens: [B, T-1, H, W] predicted frame tokens  
            factored_logits: [B, 512, 2, T-1, H, W] prediction logits
        """
        WINDOW_SIZE = 16
        from einops import rearrange
        
        # Reshape input to spatial-temporal format
        B = input_ids.shape[0]
        H = W = 16  # Fixed spatial dimensions for 256 tokens (16x16)
        inputs_THW = rearrange(input_ids, "b (t h w) -> b t h w", 
                              t=WINDOW_SIZE, h=H, w=W).to(self.device)
        
        all_samples = []
        all_logits = []
        
        with torch.no_grad():
            # Predict each frame using previous frames as context (teacher-forced)
            for timestep in range(1, WINDOW_SIZE):
                # Context: frames 0 to timestep-1
                context_frames = inputs_THW[:, :timestep]  # [B, timestep, H, W]
                context_flat = rearrange(context_frames, "b t h w -> b (t h w)")
                
                # Forward pass with context only
                outputs = self.model(context_flat)
                
                # Get logits for the last predicted position
                logits_1, logits_2 = outputs.logits[:, 0], outputs.logits[:, 1]  # [B, N, 512]
                
                # Extract logits for the last frame
                N_context = context_flat.shape[1]
                last_frame_start = N_context - H * W
                frame_logits_1 = logits_1[:, last_frame_start:last_frame_start + H*W]  # [B, H*W, 512]
                frame_logits_2 = logits_2[:, last_frame_start:last_frame_start + H*W]  # [B, H*W, 512]
                
                # Reshape to spatial format for GENIE compatibility
                frame_logits_1 = frame_logits_1.view(B, H, W, 512).permute(0, 3, 1, 2)  # [B, 512, H, W]
                frame_logits_2 = frame_logits_2.view(B, H, W, 512).permute(0, 3, 1, 2)  # [B, 512, H, W]
                factored_logits = torch.stack([frame_logits_1, frame_logits_2], dim=2)  # [B, 512, 2, H, W]
                
                # Get predicted tokens
                pred_tokens_1 = torch.argmax(frame_logits_1, dim=1)  # [B, H, W]
                pred_tokens_2 = torch.argmax(frame_logits_2, dim=1)  # [B, H, W]
                
                # Reconstruct full tokens
                from genie.factorization_utils import unfactorize_token_ids
                factored_pred = torch.stack([pred_tokens_1, pred_tokens_2], dim=-1)  # [B, H, W, 2]
                predicted_tokens = unfactorize_token_ids(factored_pred, 2, 512)  # [B, H, W]
                
                all_samples.append(predicted_tokens)
                all_logits.append(factored_logits)
        
        # Stack results to match GENIE format
        samples_THW = torch.stack(all_samples, dim=1)  # [B, T-1, H, W]
        factored_logits_final = torch.stack(all_logits, dim=3)  # [B, 512, 2, T-1, H, W]
        
        return samples_THW, factored_logits_final