import json
import math
import os
import random
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset as TorchDataset

from genie.config import GenieConfig
from genie.factorization_utils import factorize_token_ids, unfactorize_token_ids
from genie.st_mask_git import cosine_schedule


class RawTokenDataset(TorchDataset):
    """Loads raw uint32 tokens as memmap-backed array"""

    def __init__(
        self,
        data_dir,
        window_size,
        stride=1,
        filter_interrupts=True,
        filter_overlaps=False,
    ):
        """
        Args:
            data_dir: directory with the same format as `data/train_v0` and `data/val_v0`.
                Notably, has `video.bin` and `metadata.json`
            window_size: number of frames per "video" sequence
            stride: frame skip
            filter_interrupts: Under 3% of training frame sequences are the concatenation of two different clips.
                If filter_interrupts is True, will filter out these sequences using the segment ids.
            filter_overlaps: If False (default), one frame will appear in multiple examples;
                e.g. frame 0 might appear as the first frame in example 0 and also the second frame in example 15.
                If True, will filter out examples so that each frame appears at most once in the dataset.
        """
        data_dir = Path(data_dir)
        with open(data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        shape = (self.metadata["num_images"], self.metadata["s"], self.metadata["s"])
        video_tokens_path, segment_ids_path, action_tokens_path = [
            data_dir / f"{name}.bin" for name in ["video", "segment_ids", "actions"]
        ]
        token_dtype = np.dtype(self.metadata.get("token_dtype", "uint32"))
        self.data = np.memmap(
            video_tokens_path, dtype=token_dtype, mode="r", shape=shape
        )
        # self.actions = np.memmap(action_tokens_path, dtype=np.uint16, mode="r", shape=(self.metadata["num_images"],))

        if os.path.isfile(segment_ids_path):
            self.segment_ids = np.memmap(
                segment_ids_path,
                dtype=np.int32,
                mode="r",
                shape=(self.metadata["num_images"],),
            )
        else:
            self.segment_ids = None
            if filter_interrupts:
                raise NotImplementedError(
                    "Cannot filter interrupted sequences without segment ids."
                )

        self.window_size, self.stride = window_size, stride
        # Number of frames between the first and last frames of a video sequence (excluding one endpoint frame)
        self.video_len = (self.window_size - 1) * self.stride

        self.valid_start_inds = []
        for start_ind in range(len(self.data) - self.video_len):
            # Assuming `segment_ids` is monotonically increasing, a sequence is interrupted
            # if the first and last frames have different segment ids.
            if not (
                filter_interrupts
                and self.segment_ids[start_ind]
                != self.segment_ids[start_ind + self.video_len]
            ):
                self.valid_start_inds.append(start_ind)

        if filter_overlaps:
            # Instead of using a sliding window, use each frame at most once
            filtered_start_inds = []
            for start_ind in self.valid_start_inds:
                overlapping_start_inds = {
                    start_ind - i * self.stride for i in range(1, self.window_size)
                }
                # all sequences from `overlapping_start_inds` will also contain `start_ind`,
                # so exclude sequence starting from `start_ind` if any of `overlapping_start_inds` is already being used
                for existing_start_ind in filtered_start_inds[
                    -self.window_size * self.stride :
                ]:
                    # Bound could be improved
                    if existing_start_ind in overlapping_start_inds:
                        break
                else:
                    filtered_start_inds.append(start_ind)

            self.valid_start_inds = filtered_start_inds

    def __len__(self):
        return len(self.valid_start_inds)

    def __getitem__(self, idx):
        """
        Returns a flattened sequence of tokens representing `self.window_size` frames,
        spaced `self.stride` apart.
        """
        start_ind = self.valid_start_inds[idx]
        x = torch.from_numpy(
            (
                self.data[start_ind : start_ind + self.video_len + 1 : self.stride]
            ).astype(np.int64)
        )
        x = x.flatten()

        attention_mask = torch.ones_like(x)
        return {
            "input_ids": x,
            "labels": x,
            "attention_mask": attention_mask,
        }


def get_maskgit_collator(config: GenieConfig):
    mask_token_id = config.image_vocab_size
    h = w = math.isqrt(config.S)

    def collate_fn(features) -> dict[str, torch.Tensor]:
        # during training, map (z_0, z_1', z_2') -> (null, z_1, z_2)
        # (z_0, z_1') -> (null, z_1) is the diffusion operator on z_1' -> z_1

        input_ids = torch.stack([ex["input_ids"] for ex in features])
        device = input_ids.device
        x_THW = rearrange(
            input_ids, "b (t h w) -> b t h w", b=len(features), t=config.T, h=h, w=w
        )
        x_THWC = factorize_token_ids(
            x_THW, config.num_factored_vocabs, config.factored_vocab_size
        )
        labels = x_THW.clone()

        # As done in Copilot-4D paper, add random noise sampled with a random rate between 0% and `config.max_corrupt_rate`
        r = torch.rand(x_THWC.size(), device=device)
        u01 = torch.rand((), device=device)
        random_patches_mask = r < config.max_corrupt_rate * u01
        random_values = torch.randint(
            low=0,
            high=config.factored_vocab_size,
            size=x_THWC.size(),
            dtype=torch.long,
            device=device,
        )
        x_THWC[random_patches_mask] = random_values[random_patches_mask]

        if random.random() < config.non_mlm_ratio:  # Closer to autoregressive inference
            # Leave frames [0, first_masked_frame) unmasked.
            first_masked_frame = random.randint(config.num_prompt_frames, config.T - 1)
            x_THWC_view = x_THWC[:, first_masked_frame:]

            # Arbitrary numbers here, but corrupting later frames more
            # since we likely have compounding errors.
            correct_rate = random.uniform(0.25, 1.0)
            for i in range(x_THWC_view.size(1)):
                correct_rate *= random.uniform(0.9, 1.0)
                r = torch.rand(
                    (len(features), h, w, config.num_factored_vocabs), device=device
                )
                random_patches_mask = r > correct_rate
                x_THWC_view[:, i][random_patches_mask] = random_values[
                    :, first_masked_frame + i
                ][random_patches_mask]
        else:  # Typical MLM masking
            first_masked_frame = 1

        mask = torch.zeros(1)
        c = 0
        while mask.max() == 0:  # We could get unlucky and mask no tokens?
            # per-minibatch, per-frame masking probability (could try variable masking rate from MUSE)
            mask_prob_T = cosine_schedule(
                torch.rand(len(features), config.T - first_masked_frame, 1, 1)
            )

            r = torch.rand_like(x_THW[:, first_masked_frame:], dtype=torch.float)
            mask = r < mask_prob_T
            c += 1

        if c > 1:
            print(f"Generated mask {c} > 1 times.")

        x_THW = unfactorize_token_ids(
            x_THWC, config.num_factored_vocabs, config.factored_vocab_size
        )
        x_THW[:, first_masked_frame:][mask] = mask_token_id

        return {
            "input_ids": rearrange(x_THW, "b t h w -> b (t h w)"),
            "labels": rearrange(labels, "b t h w -> b (t h w)"),
        }

    return collate_fn


class RawVideoDataset(TorchDataset):
    """Loads raw MP4 video files and converts them to tensors for VJEPA training"""

    def __init__(
        self,
        data_dir: Union[str, Path],
        window_size: int,
        stride: int = 1,
        filter_interrupts: bool = True,
        filter_overlaps: bool = False,
        image_size: int = 256,  # Target image size for V-JEPA
        fps: Optional[int] = None,  # If None, use original FPS
    ):
        """
        Args:
            data_dir: directory containing video_{shard}.mp4 files and metadata_{shard}.json
            window_size: number of frames per sequence
            stride: frame skip
            filter_interrupts: filter out sequences that span multiple segments
            filter_overlaps: filter out overlapping sequences
            image_size: resize frames to this size (V-JEPA typically uses 224 or 256)
            fps: target fps, if None uses original fps
        """
        data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.image_size = image_size
        self.fps = fps

        # Find all video and metadata files
        self.video_files = sorted(data_dir.glob("video_*.mp4"))
        self.metadata_files = sorted(data_dir.glob("metadata_*.json"))
        self.segment_files = sorted(data_dir.glob("segment_idx_*.bin"))

        if not self.video_files:
            raise FileNotFoundError(f"No video_*.mp4 files found in {data_dir}")

        # Load metadata for each shard
        self.shard_metadata = []
        for metadata_file in self.metadata_files:
            with open(metadata_file) as f:
                self.shard_metadata.append(json.load(f))

        # Load segment indices if available
        self.segment_indices = []
        if self.segment_files and filter_interrupts:
            for segment_file in self.segment_files:
                segments = np.fromfile(segment_file, dtype=np.int32)
                self.segment_indices.append(segments)
        elif filter_interrupts:
            raise NotImplementedError(
                "Cannot filter interrupted sequences without segment indices."
            )

        # Build frame index: (shard_idx, frame_idx)
        self.frame_index = []
        for shard_idx, video_file in enumerate(self.video_files):
            # Get number of frames in this video
            cap = cv2.VideoCapture(str(video_file))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            for frame_idx in range(frame_count):
                self.frame_index.append((shard_idx, frame_idx))

        # Number of frames between first and last frame of sequence
        self.video_len = (self.window_size - 1) * self.stride

        # Find valid starting indices
        self.valid_start_inds = []
        for start_idx in range(len(self.frame_index) - self.video_len):
            start_shard, start_frame = self.frame_index[start_idx]
            end_shard, end_frame = self.frame_index[start_idx + self.video_len]

            # Check if sequence spans multiple video files
            if start_shard != end_shard:
                continue

            # Check for segment interrupts if enabled
            if filter_interrupts and self.segment_indices:
                start_segment = self.segment_indices[start_shard][start_frame]
                end_segment = self.segment_indices[start_shard][end_frame]
                if start_segment != end_segment:
                    continue

            self.valid_start_inds.append(start_idx)

        # Filter overlaps if requested
        if filter_overlaps:
            filtered_start_inds = []
            for start_ind in self.valid_start_inds:
                overlapping_start_inds = {
                    start_ind - i * self.stride for i in range(1, self.window_size)
                }

                # Check if any overlapping sequence is already included
                for existing_start_ind in filtered_start_inds[
                    -self.window_size * self.stride :
                ]:
                    if existing_start_ind in overlapping_start_inds:
                        break
                else:
                    filtered_start_inds.append(start_ind)

            self.valid_start_inds = filtered_start_inds

        # Image preprocessing for V-JEPA
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # ImageNet normalization
            ]
        )

        print(
            f"Loaded {len(self.video_files)} video files with {len(self.valid_start_inds)} valid sequences"
        )

    def __len__(self):
        return len(self.valid_start_inds)

    def __getitem__(self, idx):
        """
        Returns a sequence of raw video frames as tensors
        Returns:
            dict with:
            - pixel_values: [T, C, H, W] tensor of video frames
            - attention_mask: [T] tensor of ones (all frames valid)
        """
        start_idx = self.valid_start_inds[idx]

        # Get frame indices for this sequence
        frame_indices = [start_idx + i * self.stride for i in range(self.window_size)]

        frames = []
        shard_idx = None
        cap = None

        try:
            for frame_idx in frame_indices:
                current_shard, current_frame = self.frame_index[frame_idx]

                # Open new video file if shard changed
                if current_shard != shard_idx:
                    if cap is not None:
                        cap.release()

                    shard_idx = current_shard
                    video_path = self.video_files[shard_idx]
                    cap = cv2.VideoCapture(str(video_path))

                # Seek to frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()

                if not ret:
                    raise RuntimeError(
                        f"Failed to read frame {current_frame} from {self.video_files[shard_idx]}"
                    )

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image and apply transforms
                pil_frame = Image.fromarray(frame)
                tensor_frame = self.transform(pil_frame)
                frames.append(tensor_frame)

        finally:
            if cap is not None:
                cap.release()

        # Stack frames: [T, C, H, W]
        pixel_values = torch.stack(frames)
        attention_mask = torch.ones(self.window_size)

        return {
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
        }


def get_raw_video_collator():
    """Simple collator for raw video data"""

    def collate_fn(features):
        pixel_values = torch.stack([ex["pixel_values"] for ex in features])
        attention_mask = torch.stack([ex["attention_mask"] for ex in features])

        return {
            "pixel_values": pixel_values,  # [B, T, C, H, W]
            "attention_mask": attention_mask,  # [B, T]
        }

    return collate_fn


class VJEPAContinuousDataset(TorchDataset):
    """Dataset for loading precomputed V-JEPA continuous embeddings"""

    def __init__(
        self,
        data_dir: Union[str, Path],
        window_size: int,
        stride: int = 1,
        filter_interrupts: bool = True,
        filter_overlaps: bool = False,
    ):
        """
        Args:
            data_dir: directory containing vjepa_embeddings.pt and actions.bin
            window_size: number of frames per sequence (for action tokens)
            stride: frame skip (for action tokens)
            filter_interrupts: filter out sequences spanning segments
            filter_overlaps: filter out overlapping sequences
        """
        data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride

        # Load metadata
        with open(data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        # Load precomputed V-JEPA embeddings
        embeddings_file = data_dir / "vjepa_embeddings.pt"
        if not embeddings_file.exists():
            raise FileNotFoundError(f"V-JEPA embeddings not found at {embeddings_file}")

        embedding_data = torch.load(embeddings_file, map_location="cpu")
        self.embeddings = embedding_data[
            "embeddings"
        ]  # [N_sequences, N_patches, embed_dim]

        # Load action tokens
        actions_file = data_dir / "action_tokens.bin"
        if actions_file.exists():
            # Read actions from binary file
            with open(actions_file, "rb") as f:
                num_sequences = int.from_bytes(f.read(4), "little")
                actions_per_sequence = int.from_bytes(f.read(4), "little")

                actions_data = np.frombuffer(f.read(), dtype=np.uint16)
                self.actions = actions_data.reshape(num_sequences, actions_per_sequence)
        else:
            # Create dummy actions if not available
            num_sequences = self.embeddings.shape[0]
            self.actions = np.zeros((num_sequences, window_size), dtype=np.uint16)

        print(f"Loaded V-JEPA continuous dataset:")
        print(f"  Embeddings: {self.embeddings.shape}")
        print(f"  Actions: {self.actions.shape}")
        print(f"  Embed dim: {self.embeddings.shape[2]}")

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        """
        Returns precomputed V-JEPA embeddings and action tokens

        Returns:
            dict with:
            - input_embeds: [N_patches, embed_dim] continuous V-JEPA embeddings
            - action_tokens: [T] action token sequence
            - attention_mask: [N_patches] mask (all ones for continuous embeddings)
        """
        # Get precomputed embeddings
        input_embeds = self.embeddings[idx]  # [N_patches, embed_dim]

        # Get action tokens
        action_tokens = torch.from_numpy(self.actions[idx].astype(np.int64))  # [T]

        # Create attention mask (all patches are valid)
        attention_mask = torch.ones(input_embeds.shape[0])  # [N_patches]

        return {
            "input_embeds": input_embeds,
            "action_tokens": action_tokens,
            "attention_mask": attention_mask,
        }


def get_vjepa_continuous_collator():
    """
    Simple collator for precomputed V-JEPA continuous embeddings.

    Returns:
        Collator function for continuous embeddings dataset
    """

    def collate_fn(features):
        # Handle variable length sequences by padding
        max_patches = max(ex["input_embeds"].size(0) for ex in features)
        max_actions = max(ex["action_tokens"].size(0) for ex in features)
        embed_dim = features[0]["input_embeds"].size(1)

        batch_size = len(features)

        # Pad input embeddings
        input_embeds = torch.zeros(batch_size, max_patches, embed_dim)
        attention_mask = torch.zeros(batch_size, max_patches, dtype=torch.bool)
        action_tokens = torch.zeros(batch_size, max_actions, dtype=torch.long)

        for i, ex in enumerate(features):
            seq_len = ex["input_embeds"].size(0)
            action_len = ex["action_tokens"].size(0)

            input_embeds[i, :seq_len] = ex["input_embeds"]
            attention_mask[i, :seq_len] = ex.get(
                "attention_mask", torch.ones(seq_len, dtype=torch.bool)
            )
            action_tokens[i, :action_len] = ex["action_tokens"]

        return {
            "input_embeds": input_embeds,  # [B, N_patches, embed_dim]
            "action_tokens": action_tokens,  # [B, T]
            "attention_mask": attention_mask,  # [B, N_patches]
        }

    return collate_fn
