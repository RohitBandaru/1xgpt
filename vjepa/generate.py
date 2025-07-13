"""
V-JEPA generation script - equivalent to genie/generate.py

Example usage:
python vjepa/generate.py --checkpoint_dir data/vjepa_model/final_checkpt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator

# 1xgpt imports
sys.path.append(os.getcwd())
from data import RawTokenDataset
from vjepa.model import VJEPAWorldModel


def parse_args():
    parser = argparse.ArgumentParser(description="Generate frames using V-JEPA model.")
    parser.add_argument(
        "--checkpoint_dir", type=str, required=True,
        help="Path to V-JEPA model checkpoint directory."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/vjepa_generated",
        help="Directory to save generated tokens."
    )
    parser.add_argument(
        "--val_data_dir", type=str, default="data/val_v1.1",
        help="Directory containing validation data."
    )
    parser.add_argument(
        "--example_ind", type=int, default=0,
        help="Index of example to generate from."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for generation."
    )
    parser.add_argument(
        "--num_prompt_frames", type=int, default=8,
        help="Number of prompt frames for generation."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for generation."
    )
    
    return parser.parse_args()


class VJEPAGenerator:
    """V-JEPA frame generator - equivalent to GenieEvaluator"""
    
    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        self.device = device
        
        # Load V-JEPA model
        self.model = VJEPAWorldModel.from_pretrained(checkpoint_dir)
        self.model = self.model.to(device)
        self.model.eval()
        
    def generate_frames(self, input_ids: torch.LongTensor, action_tokens=None) -> torch.LongTensor:
        """
        Generate future frames autoregressively.
        
        Args:
            input_ids: [B, T*H*W] input token sequence
            action_tokens: [B, T] action tokens (optional)
            
        Returns:
            generated_tokens: [B, T, H, W] generated frame tokens
        """
        with torch.no_grad():
            # TODO: Implement autoregressive generation
            # For now, just run forward pass
            outputs = self.model(input_ids, action_tokens)
            
            # Convert logits to tokens
            logits_1, logits_2 = outputs.logits[:, 0], outputs.logits[:, 1]
            tokens_1 = torch.argmax(logits_1, dim=-1)
            tokens_2 = torch.argmax(logits_2, dim=-1)
            
            # Reconstruct tokens from factorized representation
            from genie.factorization_utils import unfactorize_token_ids
            B, N = tokens_1.shape
            T, H, W = 16, 16, 16  # TODO: Get from config
            
            factored_tokens = torch.stack([tokens_1, tokens_2], dim=-1)  # [B, N, 2]
            generated_tokens = unfactorize_token_ids(factored_tokens, 2, 512)  # [B, N]
            generated_tokens = generated_tokens.view(B, T, H, W)
            
            return generated_tokens


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load validation dataset
    val_dataset = RawTokenDataset(
        args.val_data_dir, 
        window_size=16, 
        stride=15, 
        filter_overlaps=True
    )
    
    # Create data loader
    dataloader = DataLoader(
        val_dataset, 
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Initialize generator
    generator = VJEPAGenerator(args.checkpoint_dir, args.device)
    
    print(f"Generating from example {args.example_ind}")
    
    # Get specific example or first batch
    for i, batch in enumerate(dataloader):
        if i * args.batch_size <= args.example_ind < (i + 1) * args.batch_size:
            batch_idx = args.example_ind - i * args.batch_size
            
            input_ids = batch["input_ids"][batch_idx:batch_idx+1].to(args.device)
            
            # Generate frames
            generated_tokens = generator.generate_frames(input_ids)
            
            # Save generated tokens in same format as GENIE
            # Flatten to match expected format for visualize.py
            flattened_tokens = generated_tokens.flatten(1)  # [B, T*H*W]
            
            # Create video.bin and metadata.json files
            video_path = Path(args.output_dir) / "video.bin"
            metadata_path = Path(args.output_dir) / "metadata.json"
            
            # Save tokens as memmap-compatible format
            flattened_tokens.numpy().astype(np.uint32).tofile(video_path)
            
            # Create metadata compatible with visualize.py
            T, H, W = generated_tokens.shape[1:]
            metadata = {
                "num_images": T,
                "s": H,  # spatial dimension
                "vocab_size": 262144,  # MAGVIT2 vocab size
                "hz": 2,  # frame rate
                "num_prompt_frames": args.num_prompt_frames,
                "window_size": T,
                "token_dtype": "uint32"
            }
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Generated tokens saved to {args.output_dir}")
            print(f"Shape: {generated_tokens.shape}")
            print(f"Use: python visualize.py --token_dir {args.output_dir}")
            break
    else:
        print(f"Example {args.example_ind} not found in dataset")


if __name__ == "__main__":
    main()