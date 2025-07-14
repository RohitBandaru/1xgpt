"""
V-JEPA evaluation script - equivalent to genie/evaluate.py

Example usage:
python vjepa/evaluate.py --checkpoint_dir data/vjepa_model/final_checkpt
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import lpips
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator

# 1xgpt imports
sys.path.append(os.getcwd())
from data import RawTokenDataset
from visualize import decode_latents_wrapper
from eval_utils import decode_tokens, compute_lpips, AvgMetric, compute_loss
from vjepa.model import VJEPAWorldModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate V-JEPA model.")
    parser.add_argument(
        "--val_data_dir", type=str, default="data/val_v1.1",
        help="Directory containing validation data."
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to V-JEPA model checkpoint."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--max_examples", type=int, default=None,
        help="Maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--save_outputs_dir", type=str, default=None,
        help="Directory to save evaluation outputs."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation."
    )
    parser.add_argument(
        "--skip_lpips", action="store_true",
        help="Skip LPIPS computation to speed up evaluation."
    )
    
    return parser.parse_args()


class VJEPAEvaluator:
    """V-JEPA evaluator - equivalent to GenieEvaluator"""
    
    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        self.device = device
        
        # Load V-JEPA model
        self.model = VJEPAWorldModel.from_pretrained(checkpoint_dir)
        self.model = self.model.to(device)
        self.model.eval()
        
    def predict_next_frames(self, input_ids: torch.LongTensor, action_tokens=None):
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


@torch.no_grad()
def main():
    args = parse_args()
    
    # Load validation dataset
    val_dataset = RawTokenDataset(
        args.val_data_dir, 
        window_size=16, 
        stride=15, 
        filter_overlaps=True
    )
    
    if args.max_examples is not None:
        val_dataset.valid_start_inds = val_dataset.valid_start_inds[:args.max_examples]
    
    dataloader = DataLoader(
        val_dataset, 
        collate_fn=default_data_collator, 
        batch_size=args.batch_size
    )
    
    # Initialize evaluator and metrics
    evaluator = VJEPAEvaluator(args.checkpoint_dir, args.device)
    decode_latents = decode_latents_wrapper()
    lpips_alex = lpips.LPIPS(net="alex")
    metrics = defaultdict(AvgMetric)
    
    if args.save_outputs_dir is not None:
        os.makedirs(args.save_outputs_dir, exist_ok=True)
        outputs_to_save = defaultdict(list)
    
    print("Starting V-JEPA evaluation...")
    
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch_size = batch["input_ids"].size(0)
        
        # Move to device
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        
        # Predict next frames
        start_time = time.time()
        predicted_tokens, factored_logits = evaluator.predict_next_frames(input_ids)
        gen_time = (time.time() - start_time) / (15 * batch_size)  # 15 predicted frames
        metrics["gen_time"].update(gen_time, batch_size)
        
        # Compute loss using 1X challenge metric
        loss = compute_loss(labels, factored_logits)
        metrics["loss"].update(loss, batch_size)
        
        # Compute accuracy
        reshaped_input_ids = input_ids.view(batch_size, 16, 16, 16)[:, 1:]  # Skip first frame
        acc = (reshaped_input_ids.to(args.device) == predicted_tokens).float().mean().item()
        metrics["acc"].update(acc, batch_size)
        
        # Decode frames for LPIPS calculation
        if not args.skip_lpips:
            start_time = time.time()
            pred_frames = decode_tokens(predicted_tokens.cpu(), decode_latents)
            decoded_gtruth = decode_tokens(reshaped_input_ids.cpu(), decode_latents)
            dec_time = (time.time() - start_time) / (15 * batch_size)
            metrics["dec_time"].update(dec_time, batch_size)
            
            # Compute LPIPS
            lpips_scores = compute_lpips(decoded_gtruth, pred_frames, lpips_alex)
            metrics["pred_lpips"].update_list(lpips_scores)
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}: {dict((key, f'{val.mean():.4f}') for key, val in metrics.items())}")
        
        # Save outputs if requested
        if args.save_outputs_dir is not None and not args.skip_lpips:
            outputs_to_save["pred_frames"].append(pred_frames)
            outputs_to_save["pred_logits"].append(factored_logits.cpu())
            outputs_to_save["gtruth_frames"].append(decoded_gtruth)
            outputs_to_save["gtruth_tokens"].append(reshaped_input_ids.cpu())
    
    # Final results
    print("\n" + "="*50)
    print("V-JEPA Evaluation Results:")
    print("="*50)
    for key, val in metrics.items():
        print(f"{key:15s}: {val.mean():.4f}")
    
    # Save outputs if requested
    if args.save_outputs_dir is not None and not args.skip_lpips:
        from pathlib import Path
        save_path = Path(args.save_outputs_dir)
        torch.save(torch.cat(outputs_to_save["pred_frames"], dim=0).cpu(), save_path / "pred_frames.pt")
        torch.save(torch.cat(outputs_to_save["pred_logits"], dim=0).cpu(), save_path / "pred_logits.pt")
        torch.save(torch.cat(outputs_to_save["gtruth_frames"], dim=0).cpu(), save_path / "gtruth_frames.pt")
        torch.save(torch.cat(outputs_to_save["gtruth_tokens"], dim=0).cpu(), save_path / "gtruth_tokens.pt")
        print(f"Outputs saved to {args.save_outputs_dir}")


if __name__ == "__main__":
    main()