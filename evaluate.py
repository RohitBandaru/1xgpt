"""
Shared evaluation framework for 1X GPT models.

Example usage:
python evaluate.py --model_type genie --checkpoint_dir 1x-technologies/GENIE_138M --skip_lpips
python evaluate.py --model_type vjepa --checkpoint_dir data/vjepa_model/final_checkpt --skip_lpips
"""

import argparse
import time
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import lpips
import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator

# 1xgpt imports
sys.path.append(os.getcwd())
from data import RawTokenDataset
from visualize import decode_latents_wrapper
from eval_utils import decode_tokens, compute_lpips, AvgMetric, compute_loss

# Hardcoded values for the v1.1 dataset
WINDOW_SIZE = 16
STRIDE = 15  # Data is 30 Hz so with stride 15, video is 2 Hz


class BaseEvaluator(ABC):
    """Abstract base class for model evaluators."""
    
    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.model = None
        self._load_model()
        
    @abstractmethod
    def _load_model(self):
        """Load the model from checkpoint."""
        pass
    
    @abstractmethod
    def predict_and_get_logits(self, input_ids: torch.LongTensor, **kwargs) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Predict next frames and return both samples and logits.
        
        Args:
            input_ids: LongTensor of size (B, T*H*W) corresponding to flattened, tokenized images.
            **kwargs: Model-specific arguments
            
        Returns:
            samples_THW: size (B, T-1, H, W) corresponding to the token ids of the predicted frames.
            factored_logits: size (B, 512, 2, T-1, H, W) corresponding to the predicted logits.
        """
        pass
    
    def predict_next_frames(self, samples_THW: torch.LongTensor, decode_latents) -> torch.Tensor:
        """
        Convert predicted tokens to frames.
        
        Args:
            samples_THW: LongTensor of size (B, T-1, H, W) corresponding to sampled images in the quantized latent space.
            decode_latents: Decoder function
            
        Returns:
            LongTensor of size (B, T-1, 3, 256, 256) corresponding to the predicted frames.
        """
        return decode_tokens(samples_THW.cpu(), decode_latents)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate 1X GPT models.")
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["genie", "vjepa"],
        help="Type of model to evaluate."
    )
    parser.add_argument(
        "--val_data_dir", type=str, default="data/val_v1.1",
        help="A directory with video data, should have a `metadata.json` and `video.bin`."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to a HuggingFace-style checkpoint."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size, current script only supports a single GPU."
    )
    parser.add_argument(
        "--save_outputs_dir", type=str,
        help="Debug option. If specified, will save model predictions and ground truths to this directory."
    )
    parser.add_argument(
        "--max_examples", type=int,
        help="If specified, will stop evaluation early after `max_examples` examples."
    )
    parser.add_argument(
        "--skip_lpips", action="store_true",
        help="Skip LPIPS computation to speed up evaluation (only compute loss and accuracy)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation."
    )
    
    # GENIE-specific arguments
    parser.add_argument(
        "--maskgit_steps", type=int, default=2, help="Number of MaskGIT sampling steps (GENIE only)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0,
        help="Sampling temperature (GENIE only). If `temperature` <= 1e-8, will do greedy sampling."
    )

    return parser.parse_args()


def create_evaluator(model_type: str, checkpoint_dir: str, device: str, **kwargs) -> BaseEvaluator:
    """Factory function to create the appropriate evaluator."""
    if model_type == "genie":
        from genie.evaluator import GenieEvaluator
        return GenieEvaluator(checkpoint_dir, device, **kwargs)
    elif model_type == "vjepa":
        from vjepa.evaluator import VJEPAEvaluator
        return VJEPAEvaluator(checkpoint_dir, device, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@torch.no_grad()
def main():
    transformers.set_seed(42)
    args = parse_args()

    # Load dataset
    val_dataset = RawTokenDataset(args.val_data_dir, window_size=WINDOW_SIZE, stride=STRIDE, filter_overlaps=True)
    args.latent_h = args.latent_w = val_dataset.metadata["s"]

    # Initialize components
    decode_latents = decode_latents_wrapper()
    if not args.skip_lpips:
        lpips_alex = lpips.LPIPS(net="alex")

    if args.max_examples is not None:
        val_dataset.valid_start_inds = val_dataset.valid_start_inds[:args.max_examples]

    dataloader = DataLoader(val_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

    # Create model-specific evaluator
    model_kwargs = {}
    if args.model_type == "genie":
        model_kwargs = {
            "maskgit_steps": args.maskgit_steps,
            "temperature": args.temperature,
            "latent_h": args.latent_h,
            "latent_w": args.latent_w
        }

    evaluator = create_evaluator(args.model_type, args.checkpoint_dir, args.device, **model_kwargs)
    metrics = defaultdict(AvgMetric)

    if args.save_outputs_dir is not None:
        outputs_to_save = defaultdict(list)

    print(f"Starting {args.model_type.upper()} evaluation...")
    eval_start_time = time.time()

    for batch in tqdm(dataloader):
        batch_size = batch["input_ids"].size(0)
        
        # Move to device
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)

        # Predict frames and get logits
        start_time = time.time()
        samples, factored_logits = evaluator.predict_and_get_logits(input_ids)
        frames_per_batch = (WINDOW_SIZE - 1) * batch_size
        metrics["gen_time"].update((time.time() - start_time) / frames_per_batch, batch_size)

        # Compute loss
        loss = compute_loss(labels, factored_logits)
        metrics["loss"].update(loss, batch_size)

        # Compute accuracy
        reshaped_input_ids = input_ids.view(batch_size, WINDOW_SIZE, args.latent_h, args.latent_w)
        ground_truth = reshaped_input_ids[:, 1:].to(args.device)
        acc = (ground_truth == samples).float().mean().item()
        metrics["acc"].update(acc, batch_size)

        # Only compute LPIPS if not skipped (slow image decoding)
        if not args.skip_lpips:
            start_time = time.time()
            pred_frames = evaluator.predict_next_frames(samples, decode_latents)
            metrics["dec_time"].update((time.time() - start_time) / frames_per_batch, batch_size)

            decoded_gtruth = decode_tokens(ground_truth.cpu(), decode_latents)
            metrics["pred_lpips"].update_list(compute_lpips(decoded_gtruth, pred_frames, lpips_alex))
        
        # Print progress
        print({key: f"{val.mean():.4f}" for key, val in metrics.items()})
        
        if args.save_outputs_dir is not None and not args.skip_lpips:
            outputs_to_save["pred_frames"].append(pred_frames)
            outputs_to_save["pred_logits"].append(factored_logits.cpu())
            outputs_to_save["gtruth_frames"].append(decoded_gtruth)
            outputs_to_save["gtruth_tokens"].append(ground_truth.cpu())

    # Final results
    print("\n" + "="*50)
    print(f"{args.model_type.upper()} Evaluation Results:")
    print("="*50)
    for key, val in metrics.items():
        print(f"{key:15s}: {val.mean():.4f}")

    # Save outputs if requested
    if args.save_outputs_dir is not None and not args.skip_lpips:
        os.makedirs(args.save_outputs_dir, exist_ok=True)
        save_outputs_dir = Path(args.save_outputs_dir)
        torch.save(torch.cat(outputs_to_save["pred_frames"], dim=0).cpu(), save_outputs_dir / "pred_frames.pt")
        torch.save(torch.cat(outputs_to_save["pred_logits"], dim=0).cpu(), save_outputs_dir / "pred_logits.pt")
        torch.save(torch.cat(outputs_to_save["gtruth_frames"], dim=0).cpu(), save_outputs_dir / "gtruth_frames.pt")
        torch.save(torch.cat(outputs_to_save["gtruth_tokens"], dim=0).cpu(), save_outputs_dir / "gtruth_tokens.pt")
        print(f"Outputs saved to {args.save_outputs_dir}")

    # Print total evaluation time
    total_eval_time = time.time() - eval_start_time
    print(f"\nTotal evaluation time (not including setup): {total_eval_time:.2f} seconds ({total_eval_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()