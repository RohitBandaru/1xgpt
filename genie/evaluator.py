"""
GENIE-specific evaluator implementation.

This file contains the core evaluation logic extracted from the original
genie/evaluate.py file. The original file has been removed and its functionality
has been refactored into this model-specific evaluator class and the shared
evaluation framework in ../evaluate.py.

Original file: genie/evaluate.py
Refactored: 2025-07-14
"""

import torch
from einops import rearrange
from typing import Tuple

from evaluate import BaseEvaluator
from genie.st_mask_git import STMaskGIT


class GenieEvaluator(BaseEvaluator):
    """GENIE model evaluator."""
    
    def __init__(self, checkpoint_dir: str, device: str = "cuda", 
                 maskgit_steps: int = 2, temperature: float = 0,
                 latent_h: int = 16, latent_w: int = 16):
        self.maskgit_steps = maskgit_steps
        self.temperature = temperature
        self.latent_h = latent_h
        self.latent_w = latent_w
        super().__init__(checkpoint_dir, device)
        
    def _load_model(self):
        """Load GENIE model from checkpoint."""
        self.model = STMaskGIT.from_pretrained(self.checkpoint_dir)
        self.model = self.model.to(device=self.device)
        self.model.eval()
    
    def predict_and_get_logits(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Conditioned on each prefix: [frame_0], [frame_0, frame_1], ..., [frame_0, frame_1, ... frame_{T-1}],
        predict the tokens in the following frame: [pred_frame_1, pred_frame_2, ..., pred_frame_T].

        Image logits are denoised in parallel across spatial dimension and teacher-forced
        across the time dimension. To compute logits, we save both the samples and logits as we do MaskGIT generation.

        Total number of forward passes is (T-1) * maskgit steps.

        Args:
            input_ids: LongTensor of size (B, T*H*W) corresponding to flattened, tokenized images.

        Returns: (samples_THW, factored_logits)
            samples_THW:
                size (B, T-1, H, W) corresponding to the token ids of the predicted frames.
                May differ from the argmax of `factored_logits` if not greedy sampling.
            factored_logits:
                size (B, 512, 2, T-1, H, W) corresponding to the predicted logits.
                Note that we are factorizing the 2**18 vocabulary into two separate vocabularies of size 512 each.
        """
        WINDOW_SIZE = 16
        inputs_THW = rearrange(input_ids, "b (t h w) -> b t h w", t=WINDOW_SIZE,
                               h=self.latent_h, w=self.latent_w).to(self.device)
        all_samples = []
        all_logits = []
        
        for timestep in range(1, WINDOW_SIZE):
            print(f"Generating frame {timestep}")
            inputs_masked = inputs_THW.clone()
            inputs_masked[:, timestep:] = self.model.mask_token_id

            # MaskGIT sampling
            samples_HW, factored_logits = self.model.maskgit_generate(
                inputs_masked, out_t=timestep, maskgit_steps=self.maskgit_steps,
                temperature=self.temperature,
            )

            all_samples.append(samples_HW)
            all_logits.append(factored_logits)

        samples_THW = torch.stack(all_samples, dim=1)
        return samples_THW, torch.stack(all_logits, dim=3)