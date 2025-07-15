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
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import RawVideoDataset, get_raw_video_collator
from vjepa.encoder import VJEPAEncoder, create_vjepa_encoder
from vjepa.config import VJEPAEncoderConfig


def generate_embeddings(
    input_dir: str,
    output_dir: str,
    config_path: str,
    window_size: int = 16,
    batch_size: int = 4,
    num_workers: int = 4,
    device: str = "cuda",
    analyze_embeddings: bool = True
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
    encoder = create_vjepa_encoder(config_path, add_tokenizer=True)
    encoder.to(device)
    encoder.eval()
    
    # Load dataset
    print(f"\n📂 Loading raw video dataset...")
    dataset = RawVideoDataset(
        data_dir=input_dir,
        window_size=window_size,
        stride=1
    )
    
    collate_fn = get_raw_video_collator()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
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
            action_tokens = batch["action_tokens"].to(device)  # [B, T]
            
            B, T, C, H, W = pixel_values.shape
            
            # Tokenize videos using V-JEPA encoder
            vjepa_tokens = encoder.tokenize_video(pixel_values)  # [B, N_patches]
            
            # Reshape tokens to spatial-temporal format
            H_tokens, W_tokens = H // 16, W // 16  # Assuming 16x16 patches
            N_spatial = H_tokens * W_tokens
            
            if vjepa_tokens.size(1) == T * N_spatial:
                # Perfect match - reshape directly
                vjepa_tokens_reshaped = vjepa_tokens.view(B, T, H_tokens, W_tokens)
            else:
                # Adaptive reshaping
                print(f"⚠️  Warning: N_patches={vjepa_tokens.size(1)} != T*H*W={T*N_spatial}")
                # Take first T*N_spatial tokens and reshape
                vjepa_tokens_flat = vjepa_tokens[:, :T*N_spatial]
                vjepa_tokens_reshaped = vjepa_tokens_flat.view(B, T, H_tokens, W_tokens)
            
            # Flatten to sequence format (compatible with training pipeline)
            vjepa_tokens_flat = vjepa_tokens_reshaped.view(B, T * H_tokens * W_tokens)
            
            # Store results
            all_tokens.append(vjepa_tokens_flat.cpu().numpy())
            all_action_tokens.append(action_tokens.cpu().numpy())
            
            # Collect samples for analysis
            if analyze_tokens and batch_idx < 3:  # Analyze first 3 batches
                try:
                    analysis = encoder.analyze_tokens(pixel_values)
                    analysis_samples.append(analysis)
                except Exception as e:
                    print(f"⚠️  Analysis failed for batch {batch_idx}: {e}")
            
            # Debug info for first batch
            if batch_idx == 0:
                print(f"\n📊 First batch processing:")
                print(f"  Input shape: {pixel_values.shape}")
                print(f"  V-JEPA tokens: {vjepa_tokens.shape}")
                print(f"  Reshaped tokens: {vjepa_tokens_reshaped.shape}")
                print(f"  Final tokens: {vjepa_tokens_flat.shape}")
                print(f"  Token range: [{vjepa_tokens_flat.min().item()}, {vjepa_tokens_flat.max().item()}]")
    
    # Concatenate all batches
    print(f"\n📋 Concatenating results...")
    all_tokens = np.concatenate(all_tokens, axis=0)  # [N_sequences, T*H*W]
    all_action_tokens = np.concatenate(all_action_tokens, axis=0)  # [N_sequences, T]
    
    print(f"Final dataset:")
    print(f"  V-JEPA tokens: {all_tokens.shape}")
    print(f"  Action tokens: {all_action_tokens.shape}")
    print(f"  V-JEPA token range: [{all_tokens.min()}, {all_tokens.max()}]")
    
    # Create output structure compatible with training pipeline
    train_output_dir = Path(output_dir) / "train_v1.1"
    os.makedirs(train_output_dir, exist_ok=True)
    
    # Save V-JEPA tokens
    tokens_file = train_output_dir / "image_tokens.bin"
    print(f"\n💾 Saving V-JEPA tokens to {tokens_file}")
    
    with open(tokens_file, 'wb') as f:
        # Write header (compatible with RawTokenDataset)
        f.write(all_tokens.shape[0].to_bytes(4, 'little'))  # Number of sequences
        f.write(all_tokens.shape[1].to_bytes(4, 'little'))  # Tokens per sequence
        
        # Write tokens as uint16 (V-JEPA vocab size < 65536)
        all_tokens_uint16 = all_tokens.astype(np.uint16)
        f.write(all_tokens_uint16.tobytes())
    
    # Save action tokens
    actions_file = train_output_dir / "action_tokens.bin"
    print(f"💾 Saving action tokens to {actions_file}")
    
    with open(actions_file, 'wb') as f:
        # Write header
        f.write(all_action_tokens.shape[0].to_bytes(4, 'little'))  # Number of sequences
        f.write(all_action_tokens.shape[1].to_bytes(4, 'little'))  # Actions per sequence
        
        # Write actions as uint16
        all_action_tokens_uint16 = all_action_tokens.astype(np.uint16)
        f.write(all_action_tokens_uint16.tobytes())
    
    # Create metadata compatible with RawTokenDataset
    H_tokens, W_tokens = H // 16, W // 16
    metadata = {
        "s": H_tokens,  # Spatial dimension
        "vocab_size": int(all_tokens.max() + 1),  # Actual vocabulary size used
        "hz": 30,  # Frames per second
        "total_sequences": int(all_tokens.shape[0]),
        "tokens_per_sequence": int(all_tokens.shape[1]),
        "actions_per_sequence": int(all_action_tokens.shape[1]),
        "window_size": window_size,
        "spatial_size": H_tokens * W_tokens,
        "temporal_size": T,
        "patch_size": 16,
        "created_with": "tokenize_videos.py (V-JEPA encoder)",
        "encoder_type": "vjepa"
    }
    
    metadata_file = train_output_dir / "metadata.json"
    print(f"💾 Saving metadata to {metadata_file}")
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save V-JEPA config for training
    config = VJEPAConfig.from_pretrained(config_path)
    vjepa_config_dict = config.to_dict()
    vjepa_config_dict["actual_vjepa_vocab_size"] = int(all_tokens.max() + 1)
    vjepa_config_dict["tokenized_with"] = "vjepa_encoder"
    
    config_file = output_dir / "vjepa_config.json"
    print(f"💾 Saving V-JEPA config to {config_file}")
    
    with open(config_file, 'w') as f:
        json.dump(vjepa_config_dict, f, indent=2)
    
    # Save analysis results
    if analyze_tokens and analysis_samples:
        print(f"\n📈 Token Analysis Results:")
        
        # Aggregate analysis
        total_vocab_usage = 0
        total_utilization = 0
        feature_stats_agg = {'mean': [], 'std': [], 'min': [], 'max': []}
        
        for i, analysis in enumerate(analysis_samples):
            vocab_usage = analysis['vocab_usage']
            feature_stats = analysis['feature_stats']
            
            print(f"  Batch {i+1}:")
            print(f"    Vocabulary usage: {vocab_usage['unique_tokens']}/{vocab_usage['vocab_size']} ({vocab_usage['utilization']:.1%})")
            print(f"    Token range: {vocab_usage['token_range']}")
            print(f"    Feature stats: mean={feature_stats['mean']:.3f}, std={feature_stats['std']:.3f}")
            
            total_vocab_usage += vocab_usage['unique_tokens']
            total_utilization += vocab_usage['utilization']
            
            for key in feature_stats_agg:
                feature_stats_agg[key].append(feature_stats[key])
        
        # Save detailed analysis
        analysis_file = output_dir / "tokenization_analysis.json"
        analysis_summary = {
            "samples_analyzed": len(analysis_samples),
            "avg_vocab_utilization": total_utilization / len(analysis_samples),
            "feature_stats_summary": {
                key: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
                for key, values in feature_stats_agg.items()
            },
            "detailed_samples": analysis_samples
        }
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis_summary, f, indent=2)
        
        print(f"💾 Saved detailed analysis to {analysis_file}")
    
    print(f"\n✅ V-JEPA tokenization completed successfully!")
    print(f"\n📁 Output files:")
    print(f"   📄 Tokens: {tokens_file}")
    print(f"   📄 Actions: {actions_file}")
    print(f"   📄 Metadata: {metadata_file}")
    print(f"   📄 Config: {config_file}")
    if analyze_tokens:
        print(f"   📄 Analysis: {output_dir}/tokenization_analysis.json")
    
    print(f"\n🚀 Usage:")
    print(f"   python train.py --vjepa_config {config_file} --train_data_dir {train_output_dir}")
    
    return {
        "tokens_shape": all_tokens.shape,
        "actions_shape": all_action_tokens.shape,
        "vocab_size": int(all_tokens.max() + 1),
        "output_dir": str(output_dir)
    }


def main():
    parser = argparse.ArgumentParser(description="Tokenize videos using V-JEPA encoder")
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing raw MP4 videos"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save V-JEPA tokens"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to V-JEPA config file"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Number of frames per sequence"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for computation"
    )
    parser.add_argument(
        "--no_analysis",
        action="store_true",
        help="Skip token quality analysis"
    )
    
    args = parser.parse_args()
    
    tokenize_videos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        window_size=args.window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        analyze_tokens=not args.no_analysis
    )


if __name__ == "__main__":
    main()