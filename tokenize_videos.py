#!/usr/bin/env python3
"""
Generate continuous V-JEPA embeddings from videos.

This script processes raw video files into continuous V-JEPA embeddings
that preserve the rich representations from the ViT-G model.

For discrete tokens, use the precomputed COSMOS tokens from 1X dataset.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import RawVideoDataset, get_raw_video_collator
from vjepa.config import VJEPAEncoderConfig
from vjepa.encoder import VJEPAEncoder


def generate_continuous_embeddings(
    input_dir: str,
    output_dir: str,
    config_path: str,
    window_size: int = 16,
    batch_size: int = 4,
    num_workers: int = 4,
    device: str = "cuda",
    analyze_embeddings: bool = False,
):
    """
    Generate continuous V-JEPA embeddings from raw videos.

    Args:
        input_dir: Directory containing raw MP4 videos
        output_dir: Directory to save V-JEPA embeddings
        config_path: Path to V-JEPA config file
        window_size: Number of frames per sequence
        batch_size: Batch size for processing
        num_workers: Number of data loading workers
        device: Device for computation
        analyze_embeddings: Whether to analyze embedding quality
    """
    print(f"🎬 V-JEPA Embedding Generation")
    print(f"{'='*50}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Config: {config_path}")
    print(f"Window size: {window_size}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load V-JEPA encoder
    print(f"\n📥 Loading V-JEPA encoder...")
    config = VJEPAEncoderConfig.from_pretrained(config_path)
    encoder = VJEPAEncoder(config)
    encoder.to(device)
    encoder.eval()

    # Load dataset
    print(f"\n📂 Loading raw video dataset...")
    dataset = RawVideoDataset(data_dir=input_dir, window_size=window_size, stride=1)

    collate_fn = get_raw_video_collator()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Dataset loaded: {len(dataset)} sequences")

    # Process videos and create tokens
    all_tokens = []
    all_action_tokens = []
    analysis_samples = []

    print(f"\n🔄 Processing videos...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Tokenizing videos")):
            # Move to device
            pixel_values = batch["pixel_values"].to(device)  # [B, T, C, H, W]

            # Create dummy action tokens since RawVideoDataset doesn't provide them
            B, T = pixel_values.shape[:2]
            action_tokens = torch.zeros(B, T, dtype=torch.long, device=device)  # [B, T]

            B, T, C, H, W = pixel_values.shape

            # Encode videos using V-JEPA encoder to get continuous embeddings
            vjepa_embeddings = encoder.encode_video(
                pixel_values
            )  # [B, N_patches, embed_dim]

            # V-JEPA embeddings are already in the correct format [B, N_patches, embed_dim]
            # No need to reshape - these are continuous embeddings, not discrete tokens

            # Store results
            all_tokens.append(vjepa_embeddings.cpu().numpy())
            all_action_tokens.append(action_tokens.cpu().numpy())

            # Basic embedding info (replacing removed analyze_embeddings method)
            if analyze_embeddings and batch_idx < 3:  # Analyze first 3 batches
                try:
                    # Simple analysis without the removed method
                    analysis = {
                        "input_shape": list(pixel_values.shape),
                        "embeddings_shape": list(vjepa_embeddings.shape),
                        "embed_dim": vjepa_embeddings.shape[-1],
                        "num_patches": vjepa_embeddings.shape[1],
                        "batch_stats": {
                            "mean": vjepa_embeddings.mean().item(),
                            "std": vjepa_embeddings.std().item(),
                            "min": vjepa_embeddings.min().item(),
                            "max": vjepa_embeddings.max().item(),
                        },
                    }
                    analysis_samples.append(analysis)
                except Exception as e:
                    print(f"⚠️  Analysis failed for batch {batch_idx}: {e}")

            # Debug info for first batch
            if batch_idx == 0:
                print(f"\n📊 First batch processing:")
                print(f"  Input shape: {pixel_values.shape}")
                print(f"  V-JEPA embeddings: {vjepa_embeddings.shape}")
                print(
                    f"  Embedding range: [{vjepa_embeddings.min().item():.3f}, {vjepa_embeddings.max().item():.3f}]"
                )
                print(
                    f"  Embedding mean: {vjepa_embeddings.mean().item():.3f}, std: {vjepa_embeddings.std().item():.3f}"
                )

    # Concatenate all batches
    print(f"\n📋 Concatenating results...")
    all_tokens = np.concatenate(
        all_tokens, axis=0
    )  # [N_sequences, N_patches, embed_dim]
    all_action_tokens = np.concatenate(all_action_tokens, axis=0)  # [N_sequences, T]

    print(f"Final dataset:")
    print(f"  V-JEPA embeddings: {all_tokens.shape}")
    print(f"  Action tokens: {all_action_tokens.shape}")
    print(f"  V-JEPA embedding range: [{all_tokens.min():.3f}, {all_tokens.max():.3f}]")

    # Create output structure compatible with training pipeline
    train_output_dir = Path(output_dir) / "train_v1.1"
    os.makedirs(train_output_dir, exist_ok=True)

    # Save V-JEPA embeddings as PyTorch tensor (preserves float32 precision)
    embeddings_file = train_output_dir / "vjepa_embeddings.pt"
    print(f"\n💾 Saving V-JEPA embeddings to {embeddings_file}")

    torch.save(
        {
            "embeddings": torch.from_numpy(
                all_tokens
            ),  # [N_sequences, N_patches, embed_dim]
            "shape": all_tokens.shape,
            "dtype": str(all_tokens.dtype),
            "format": "continuous_embeddings",
        },
        embeddings_file,
    )

    # Save action tokens
    actions_file = train_output_dir / "action_tokens.bin"
    print(f"💾 Saving action tokens to {actions_file}")

    with open(actions_file, "wb") as f:
        # Write header
        f.write(all_action_tokens.shape[0].to_bytes(4, "little"))  # Number of sequences
        f.write(
            all_action_tokens.shape[1].to_bytes(4, "little")
        )  # Actions per sequence

        # Write actions as uint16
        all_action_tokens_uint16 = all_action_tokens.astype(np.uint16)
        f.write(all_action_tokens_uint16.tobytes())

    # Create metadata for continuous embeddings
    metadata = {
        "total_sequences": int(all_tokens.shape[0]),
        "num_patches": int(all_tokens.shape[1]),  # N_patches
        "embed_dim": int(all_tokens.shape[2]),  # embedding dimension
        "actions_per_sequence": int(all_action_tokens.shape[1]),
        "window_size": window_size,
        "temporal_size": T,
        "patch_size": 16,
        "image_size": H,  # Original image size
        "format": "continuous_embeddings",
        "encoder_type": "vjepa_continuous",
        "created_with": "tokenize_videos.py (V-JEPA encoder)",
        "hz": 30,  # Frames per second
    }

    metadata_file = train_output_dir / "metadata.json"
    print(f"💾 Saving metadata to {metadata_file}")

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save V-JEPA config for training
    config = VJEPAEncoderConfig.from_pretrained(config_path)
    vjepa_config_dict = config.to_dict()
    vjepa_config_dict["embed_dim"] = int(all_tokens.shape[2])
    vjepa_config_dict["num_patches"] = int(all_tokens.shape[1])
    vjepa_config_dict["tokenized_with"] = "vjepa_encoder_continuous"
    vjepa_config_dict["input_mode"] = "continuous"

    config_file = Path(output_dir) / "vjepa_config.json"
    print(f"💾 Saving V-JEPA config to {config_file}")

    with open(config_file, "w") as f:
        json.dump(vjepa_config_dict, f, indent=2)

    # Save analysis results
    if analyze_embeddings and analysis_samples:
        print(f"\n📈 Embedding Analysis Results:")

        # Aggregate analysis
        embedding_stats_agg = {
            "mean": [],
            "std": [],
            "min": [],
            "max": [],
            "norm_mean": [],
            "norm_std": [],
        }

        for i, analysis in enumerate(analysis_samples):
            embedding_stats = analysis["embedding_stats"]

            print(f"  Batch {i+1}:")
            print(f"    Input shape: {analysis['input_shape']}")
            print(f"    Embeddings shape: {analysis['embeddings_shape']}")
            print(
                f"    Embedding stats: mean={embedding_stats['mean']:.3f}, std={embedding_stats['std']:.3f}"
            )
            print(
                f"    Norm stats: mean={embedding_stats['norm_mean']:.3f}, std={embedding_stats['norm_std']:.3f}"
            )

            for key in embedding_stats_agg:
                embedding_stats_agg[key].append(embedding_stats[key])

        # Save detailed analysis
        analysis_file = Path(output_dir) / "embedding_analysis.json"
        analysis_summary = {
            "samples_analyzed": len(analysis_samples),
            "embedding_stats_summary": {
                key: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
                for key, values in embedding_stats_agg.items()
            },
            "detailed_samples": analysis_samples,
        }

        with open(analysis_file, "w") as f:
            json.dump(analysis_summary, f, indent=2)

        print(f"💾 Saved detailed embedding analysis to {analysis_file}")

    print(f"\n✅ V-JEPA continuous embedding generation completed successfully!")
    print(f"\n📁 Output files:")
    print(f"   📄 Embeddings: {embeddings_file}")
    print(f"   📄 Actions: {actions_file}")
    print(f"   📄 Metadata: {metadata_file}")
    print(f"   📄 Config: {config_file}")
    if analyze_embeddings:
        print(f"   📄 Analysis: {output_dir}/embedding_analysis.json")

    print(f"\n🚀 Usage:")
    print(
        f"   python train.py --vjepa_config {config_file} --train_data_dir {train_output_dir}"
    )

    return {
        "embeddings_shape": all_tokens.shape,
        "actions_shape": all_action_tokens.shape,
        "embed_dim": int(all_tokens.shape[2]),
        "output_dir": str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Tokenize videos using V-JEPA encoder")

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw MP4 videos",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save V-JEPA tokens"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to V-JEPA config file"
    )
    parser.add_argument(
        "--window_size", type=int, default=16, help="Number of frames per sequence"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for processing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device for computation"
    )
    parser.add_argument(
        "--no_analysis", action="store_true", help="Skip token quality analysis"
    )

    args = parser.parse_args()

    generate_continuous_embeddings(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        analyze_embeddings=not args.no_analysis,
    )


if __name__ == "__main__":
    main()
