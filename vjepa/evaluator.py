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
from vjepa.model import VJEPAWorldModel


class VJEPAEvaluator(BaseEvaluator):
    """V-JEPA model evaluator."""
    
    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        super().__init__(checkpoint_dir, device)
        
    def _load_model(self):
        """Load V-JEPA model from checkpoint."""
        self.model = VJEPAWorldModel.from_pretrained(self.checkpoint_dir)
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def predict_and_get_logits(self, input_ids: torch.LongTensor, action_tokens=None) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Predict next frames and return both tokens and logits.
        
        Args:
            input_ids: [B, T*H*W] input tokens
            action_tokens: [B, T] action tokens (optional)
            
        Returns:
            predicted_tokens: [B, T-1, H, W] predicted frame tokens
            factored_logits: [B, 512, 2, T-1, H, W] prediction logits
        """
        with torch.no_grad():
            outputs = self.model(input_ids, action_tokens)
            
            # Get factorized logits
            logits_1, logits_2 = outputs.logits[:, 0], outputs.logits[:, 1]  # [B, N, 512]
            
            # Convert to expected format for compute_loss
            B, N = logits_1.shape[:2]
            T, H, W = 16, 16, 16  # TODO: Get from config
            
            # Reshape to match genie format: [B, 512, 2, T-1, H, W]
            logits_1 = logits_1.view(B, T, H, W, 512).permute(0, 4, 1, 2, 3)[:, :, 1:]  # Skip first frame
            logits_2 = logits_2.view(B, T, H, W, 512).permute(0, 4, 1, 2, 3)[:, :, 1:]  # Skip first frame
            factored_logits = torch.stack([logits_1, logits_2], dim=2)  # [B, 512, 2, T-1, H, W]
            
            # Get predicted tokens
            pred_tokens_1 = torch.argmax(logits_1, dim=1)  # [B, T-1, H, W]
            pred_tokens_2 = torch.argmax(logits_2, dim=1)  # [B, T-1, H, W]
            
            # Reconstruct full tokens
            from genie.factorization_utils import unfactorize_token_ids
            factored_pred = torch.stack([pred_tokens_1, pred_tokens_2], dim=-1)  # [B, T-1, H, W, 2]
            predicted_tokens = unfactorize_token_ids(factored_pred, 2, 512)  # [B, T-1, H, W]
            
            return predicted_tokens, factored_logits