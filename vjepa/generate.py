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
from genie.factorization_utils import unfactorize_token_ids
from data import RawTokenDataset
from vjepa.predictor import VJEPAPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Generate frames using V-JEPA model.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to V-JEPA model checkpoint directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/vjepa_generated",
        help="Directory to save generated tokens.",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default="data/val_v1.1",
        help="Directory containing validation data.",
    )
    parser.add_argument(
        "--example_ind", type=int, default=0, help="Index of example to generate from."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for generation."
    )
    parser.add_argument(
        "--num_prompt_frames",
        type=int,
        default=8,
        help="Number of prompt frames for generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for generation.",
    )

    return parser.parse_args()


class VJEPAGenerator:
    """V-JEPA frame generator - equivalent to GenieEvaluator"""

    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        self.device = device

        # Load V-JEPA model
        self.model = VJEPAPredictor.from_pretrained(checkpoint_dir)
        self.model = self.model.to(device)
        self.model.eval()

    def generate_frames(
        self,
        input_ids: torch.LongTensor,
        num_prompt_frames: int = 8,
        teacher_force_time: bool = False,
        action_tokens=None,
    ) -> torch.LongTensor:
        """
        Generate future frames to match GENIE's generation interface.

        Args:
            input_ids: [B, T*H*W] input token sequence
            num_prompt_frames: Number of context frames
            teacher_force_time: Whether to use teacher forcing
            action_tokens: [B, T] action tokens (optional)

        Returns:
            generated_tokens: [B, T, H, W] generated frame tokens
        """
        from einops import rearrange

        B = input_ids.shape[0]
        T = 16  # Fixed for 1X challenge
        H = W = 16  # Fixed spatial dimensions for 256 tokens (16x16)

        # Reshape to spatial-temporal format
        example_THW = rearrange(input_ids, "b (t h w) -> b t h w", t=T, h=H, w=W)

        samples = []
        prompt_THW = example_THW.clone()

        with torch.no_grad():
            for timestep in range(num_prompt_frames, T):
                if teacher_force_time:
                    # Use ground truth up to current timestep
                    prompt_THW = example_THW.clone()

                # Get context up to current timestep
                context_frames = prompt_THW[:, :timestep]
                context_flat = rearrange(context_frames, "b t h w -> b (t h w)")

                # Forward pass to predict next frame
                outputs = self.model(context_flat, action_tokens)

                # Get logits for last predicted frame
                logits_1, logits_2 = outputs.logits[:, 0], outputs.logits[:, 1]

                # Extract logits for the frame being predicted
                N_context = context_flat.shape[1]
                last_frame_start = N_context - H * W
                frame_logits_1 = logits_1[
                    :, last_frame_start : last_frame_start + H * W
                ]
                frame_logits_2 = logits_2[
                    :, last_frame_start : last_frame_start + H * W
                ]

                # Sample next frame
                pred_tokens_1 = torch.argmax(frame_logits_1, dim=-1).view(B, H, W)
                pred_tokens_2 = torch.argmax(frame_logits_2, dim=-1).view(B, H, W)

                # Reconstruct full tokens
                factored_pred = torch.stack([pred_tokens_1, pred_tokens_2], dim=-1)
                samples_HW = unfactorize_token_ids(factored_pred, 2, 512)

                samples.append(samples_HW)

                if not teacher_force_time:
                    # Autoregressive: use prediction for next timestep
                    prompt_THW[:, timestep] = samples_HW

            # Combine prompt frames with generated frames
            generated_frames = torch.stack(
                samples, dim=1
            )  # [B, T-num_prompt_frames, H, W]
            full_sequence = torch.cat(
                [example_THW[:, :num_prompt_frames], generated_frames], dim=1
            )

            return full_sequence


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load validation dataset
    val_dataset = RawTokenDataset(
        args.val_data_dir, window_size=16, stride=15, filter_overlaps=True
    )

    # Create data loader
    dataloader = DataLoader(
        val_dataset,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Initialize generator
    generator = VJEPAGenerator(args.checkpoint_dir, args.device)

    print(f"Generating from example {args.example_ind}")

    # Get specific example or first batch
    for i, batch in enumerate(dataloader):
        if i * args.batch_size <= args.example_ind < (i + 1) * args.batch_size:
            batch_idx = args.example_ind - i * args.batch_size

            input_ids = batch["input_ids"][batch_idx : batch_idx + 1].to(args.device)

            # Generate frames
            generated_tokens = generator.generate_frames(input_ids)

            # Save generated tokens in same format as GENIE
            # Flatten to match expected format for visualize.py
            flattened_tokens = generated_tokens.flatten(1)  # [B, T*H*W]

            # Create video.bin and metadata.json files
            video_path = Path(args.output_dir) / "video.bin"
            metadata_path = Path(args.output_dir) / "metadata.json"

            # Save tokens as memmap-compatible format
            flattened_tokens.cpu().numpy().astype(np.uint32).tofile(video_path)

            # Create metadata compatible with visualize.py
            T, H, W = generated_tokens.shape[1:]
            metadata = {
                "num_images": T,
                "s": H,  # spatial dimension
                "vocab_size": 262144,  # MAGVIT2 vocab size
                "hz": 2,  # frame rate
                "num_prompt_frames": args.num_prompt_frames,
                "window_size": T,
                "token_dtype": "uint32",
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
