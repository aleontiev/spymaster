"""
Masked Gated Policy training module.

Contains:
- Embedding pre-computation for multi-scale JEPAs
- Dataset with scheduled masking for morning trading simulation
- Training and validation loops
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.model.lejepa import LeJEPA
from src.model.masked_gated_policy import MaskedGatedPolicy, DirectPolicy, MultiScalePolicy
from src.model.loss import FocalLoss
from src.data.dag.loader import load_normalized_data, RAW_CACHE_DIR, parse_date
from src.managers.checkpoint_manager import CheckpointManager

from train.common import set_seed, get_device, filter_files_by_date
from train.visualization import generate_training_charts, save_training_history


# =============================================================================
# CLI Arguments
# =============================================================================


def add_masked_policy_args(parser: argparse.ArgumentParser) -> None:
    """Add masked gated policy-specific arguments to the parser."""
    # JEPA checkpoint paths
    parser.add_argument(
        "--short-jepa-checkpoint",
        type=str,
        default=None,
        help="Path to short-term JEPA checkpoint (e.g., 15:5)",
    )
    parser.add_argument(
        "--med-jepa-checkpoint",
        type=str,
        default=None,
        help="Path to medium-term JEPA checkpoint (e.g., 45:15)",
    )
    parser.add_argument(
        "--long-jepa-checkpoint",
        type=str,
        default=None,
        help="Path to long-term JEPA checkpoint (e.g., 90:30)",
    )

    # Embeddings directory
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="data/embeddings",
        help="Directory for pre-computed embeddings",
    )
    parser.add_argument(
        "--precompute-only",
        action="store_true",
        help="Only pre-compute embeddings, don't train",
    )

    # Masking schedule
    parser.add_argument(
        "--mask-schedule",
        type=str,
        default="realistic",
        choices=["realistic", "uniform", "none"],
        help="Mask schedule for training (realistic=time-based, uniform=random)",
    )

    # Labeling parameters
    parser.add_argument(
        "--lookahead",
        type=int,
        default=15,
        help="Number of minutes to look ahead for labeling",
    )
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=0.15,
        help="Percentage threshold for directional moves",
    )

    # Model architecture
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for policy network",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability",
    )

    # Loss function
    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        help="Use focal loss instead of cross-entropy",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter",
    )
    parser.add_argument(
        "--binary-mode",
        action="store_true",
        help="Binary classification: Long vs Short only (exclude Neutral)",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help="Class weights as comma-separated values (e.g., '1.0,3.0' for binary, '1.0,4.0,1.5' for 3-class). Use 'auto' for inverse frequency weighting.",
    )
    parser.add_argument(
        "--direct-policy",
        action="store_true",
        help="Use DirectPolicy (simple MLP) instead of MaskedGatedPolicy (with attention)",
    )
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="Use MultiScalePolicy with separate heads for each scale (requires MAE labels)",
    )
    parser.add_argument(
        "--head-weight",
        type=float,
        default=1.0,
        help="Weight for individual head losses in MultiScalePolicy",
    )
    parser.add_argument(
        "--combined-weight",
        type=float,
        default=0.5,
        help="Weight for combined loss in MultiScalePolicy",
    )


# =============================================================================
# Target Position Labels
# =============================================================================


class TargetPosition:
    """Target position labels for the policy."""
    NEUTRAL = 0  # Want zero exposure
    LONG = 1     # Want to be long
    SHORT = 2    # Want to be short


def compute_target_labels(
    raw_closes: np.ndarray,
    valid_indices: List[int],
    lookahead: int = 15,
    threshold_pct: float = 0.15,
) -> np.ndarray:
    """
    Compute target position labels based on future price movement.

    Args:
        raw_closes: Array of raw close prices
        valid_indices: Indices where we have valid data
        lookahead: Number of bars to look ahead
        threshold_pct: Percentage threshold for directional moves

    Returns:
        Array of target labels (0=Neutral, 1=Long, 2=Short)
    """
    labels = []
    for idx in valid_indices:
        current_price = raw_closes[idx]
        future_price = raw_closes[idx + lookahead]

        if current_price == 0:
            labels.append(TargetPosition.NEUTRAL)
            continue

        pct_change = (future_price - current_price) / current_price * 100

        if pct_change > threshold_pct:
            labels.append(TargetPosition.LONG)
        elif pct_change < -threshold_pct:
            labels.append(TargetPosition.SHORT)
        else:
            labels.append(TargetPosition.NEUTRAL)

    return np.array(labels)


def compute_mae_label(
    raw_closes: np.ndarray,
    idx: int,
    lookahead: int,
    target_pct: float,
    max_adverse_pct: float,
) -> int:
    """
    Compute a single MAE-based label for a clean directional move.

    A move is labeled LONG/SHORT only if it reaches the target without
    excessive adverse excursion (retracement against the position).

    Args:
        raw_closes: Array of raw close prices
        idx: Current index
        lookahead: Number of bars to look ahead
        target_pct: Required return to trigger a signal (e.g., 0.15 for 0.15%)
        max_adverse_pct: Maximum allowed adverse move (e.g., 0.08 for 0.08%)

    Returns:
        Label (0=Neutral, 1=Long, 2=Short)
    """
    current_price = raw_closes[idx]

    if current_price == 0:
        return TargetPosition.NEUTRAL

    # Get the price path
    future_prices = raw_closes[idx + 1 : idx + lookahead + 1]
    if len(future_prices) < lookahead:
        return TargetPosition.NEUTRAL

    final_price = raw_closes[idx + lookahead]
    final_return = (final_price - current_price) / current_price * 100

    # Check for clean LONG: reaches target without dropping too much
    if final_return > target_pct:
        min_price = np.min(future_prices)
        worst_drawdown = (min_price - current_price) / current_price * 100
        if worst_drawdown > -max_adverse_pct:  # Drawdown is negative, so > means less severe
            return TargetPosition.LONG

    # Check for clean SHORT: reaches target without rising too much
    if final_return < -target_pct:
        max_price = np.max(future_prices)
        worst_runup = (max_price - current_price) / current_price * 100
        if worst_runup < max_adverse_pct:  # Runup is positive, so < means less severe
            return TargetPosition.SHORT

    return TargetPosition.NEUTRAL


# Default scale parameters for multi-scale labeling
SCALE_PARAMS = {
    "short": {"lookahead": 5, "target_pct": 0.08, "max_adverse_pct": 0.04},
    "medium": {"lookahead": 15, "target_pct": 0.15, "max_adverse_pct": 0.08},
    "long": {"lookahead": 30, "target_pct": 0.25, "max_adverse_pct": 0.12},
}


def compute_multiscale_mae_labels(
    raw_closes: np.ndarray,
    valid_indices: List[int],
    scale_params: Optional[Dict] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute MAE-based labels for all three scales.

    Args:
        raw_closes: Array of raw close prices
        valid_indices: Indices where we have valid data
        scale_params: Optional override for scale parameters

    Returns:
        Dictionary with 'short', 'medium', 'long' label arrays
    """
    params = scale_params or SCALE_PARAMS

    labels = {
        "short": [],
        "medium": [],
        "long": [],
    }

    for idx in valid_indices:
        for scale_name, scale_cfg in params.items():
            label = compute_mae_label(
                raw_closes,
                idx,
                lookahead=scale_cfg["lookahead"],
                target_pct=scale_cfg["target_pct"],
                max_adverse_pct=scale_cfg["max_adverse_pct"],
            )
            labels[scale_name].append(label)

    return {k: np.array(v) for k, v in labels.items()}


# =============================================================================
# Embedding Pre-computation
# =============================================================================


def precompute_embeddings(
    args: argparse.Namespace,
    device: torch.device,
) -> Path:
    """
    Pre-compute embeddings from all three JEPA models and save to disk.

    This is a one-time operation that should be run before training.
    Saves embeddings, time context, and labels to the embeddings directory.

    Args:
        args: Command line arguments with JEPA checkpoint paths
        device: Device to run computations on

    Returns:
        Path to the embeddings directory
    """
    embeddings_dir = Path(args.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("PRE-COMPUTING EMBEDDINGS")
    print("=" * 70)

    # Data directories
    data_dir = Path(args.data_dir)
    normalized_dir = data_dir / "training-1m-normalized" / args.underlying
    raw_dir = data_dir / "training-1m-raw" / args.underlying

    if not normalized_dir.exists():
        raise ValueError(f"Normalized data directory not found: {normalized_dir}")

    # Find all normalized data files
    files = sorted(normalized_dir.glob("**/*.parquet"))
    files = filter_files_by_date(files, args.start_date, args.end_date)

    if args.max_files:
        files = files[:args.max_files]

    print(f"Found {len(files)} normalized data files")

    # Get context lengths from each JEPA checkpoint
    short_ckpt = torch.load(args.short_jepa_checkpoint, map_location="cpu", weights_only=False)
    med_ckpt = torch.load(args.med_jepa_checkpoint, map_location="cpu", weights_only=False)
    long_ckpt = torch.load(args.long_jepa_checkpoint, map_location="cpu", weights_only=False)

    short_context_len = short_ckpt["config"].get("max_context_len", 15)
    med_context_len = med_ckpt["config"].get("max_context_len", 45)
    long_context_len = long_ckpt["config"].get("max_context_len", 90)

    print(f"Context lengths - Short: {short_context_len}, Med: {med_context_len}, Long: {long_context_len}")

    # Use the longest context length as the required history
    max_context_len = max(short_context_len, med_context_len, long_context_len)
    lookahead = args.lookahead

    # Collect all data
    all_data = []
    all_closes = []
    all_sin_time = []
    all_cos_time = []
    all_time_to_close = []

    print("\nLoading data files...")
    for file_path in tqdm(files, desc="Loading files"):
        try:
            # Load normalized data
            df = pd.read_parquet(file_path)
            if df is None or len(df) < max_context_len + lookahead:
                continue

            # Construct corresponding raw file path
            # normalized: training-1m-normalized/SPY/2020-01/02.parquet
            # raw: training-1m-raw/SPY/2020-01/02.parquet
            raw_file = raw_dir / file_path.parent.name / file_path.name

            if not raw_file.exists():
                raise FileNotFoundError(f"Raw data not found: {raw_file}")

            raw_df = pd.read_parquet(raw_file)
            if "close" not in raw_df.columns:
                raise ValueError(f"No 'close' column in raw data: {raw_file}")
            raw_closes = raw_df["close"].values

            all_data.append(df)
            all_closes.append(raw_closes)

            # Extract time features
            all_sin_time.append(df["sin_time"].values if "sin_time" in df.columns else np.zeros(len(df)))
            all_cos_time.append(df["cos_time"].values if "cos_time" in df.columns else np.zeros(len(df)))
            all_time_to_close.append(df["time_to_close"].values if "time_to_close" in df.columns else np.ones(len(df)))

        except Exception as e:
            print(f"  Error loading {file_path.name}: {e}")
            continue

    if not all_data:
        raise ValueError("No valid data files found!")

    print(f"\nLoaded {len(all_data)} valid days")

    # Build valid indices and concatenate data
    all_patches_data = []
    all_valid_closes = []
    all_valid_sin = []
    all_valid_cos = []
    all_valid_ttc = []
    valid_indices_per_day = []

    offset = 0
    for day_idx, (df, closes, sin_t, cos_t, ttc) in enumerate(
        zip(all_data, all_closes, all_sin_time, all_cos_time, all_time_to_close)
    ):
        day_valid_indices = []
        for i in range(max_context_len, len(df) - lookahead):
            day_valid_indices.append(i)
            all_valid_closes.append(closes[i])
            all_valid_sin.append(sin_t[i])
            all_valid_cos.append(cos_t[i])
            all_valid_ttc.append(ttc[i])

        valid_indices_per_day.append((offset, day_valid_indices, df))
        offset += len(day_valid_indices)

    total_samples = offset
    print(f"Total valid samples: {total_samples}")

    # Compute labels
    print("\nComputing target labels...")
    raw_closes_array = np.array(all_valid_closes)
    # For labeling, we need future prices - this is a simplification
    # In reality, we'd need to track the lookahead properly per day
    # For now, compute labels based on local context
    labels = []
    sample_idx = 0
    for day_offset, day_indices, df in valid_indices_per_day:
        closes = df["close"].values if "close" in df.columns else np.zeros(len(df))
        for i in day_indices:
            if i + lookahead < len(closes):
                current = closes[i]
                future = closes[i + lookahead]
                if current > 0:
                    pct_change = (future - current) / current * 100
                    if pct_change > args.threshold_pct:
                        labels.append(TargetPosition.LONG)
                    elif pct_change < -args.threshold_pct:
                        labels.append(TargetPosition.SHORT)
                    else:
                        labels.append(TargetPosition.NEUTRAL)
                else:
                    labels.append(TargetPosition.NEUTRAL)
            else:
                labels.append(TargetPosition.NEUTRAL)
            sample_idx += 1

    labels = np.array(labels)
    print(f"Label distribution: Neutral={np.sum(labels==0)}, Long={np.sum(labels==1)}, Short={np.sum(labels==2)}")

    # Compute multi-scale MAE labels
    print("\nComputing multi-scale MAE labels...")
    multiscale_labels = {"short": [], "medium": [], "long": []}
    sample_idx = 0

    for day_offset, day_indices, df in valid_indices_per_day:
        closes = df["close"].values if "close" in df.columns else np.zeros(len(df))
        for i in day_indices:
            for scale_name, scale_cfg in SCALE_PARAMS.items():
                label = compute_mae_label(
                    closes,
                    i,
                    lookahead=scale_cfg["lookahead"],
                    target_pct=scale_cfg["target_pct"],
                    max_adverse_pct=scale_cfg["max_adverse_pct"],
                )
                multiscale_labels[scale_name].append(label)
            sample_idx += 1

    # Convert to arrays
    for scale_name in multiscale_labels:
        multiscale_labels[scale_name] = np.array(multiscale_labels[scale_name])

    print("Multi-scale label distributions:")
    for scale_name, scale_labels in multiscale_labels.items():
        n_neutral = np.sum(scale_labels == 0)
        n_long = np.sum(scale_labels == 1)
        n_short = np.sum(scale_labels == 2)
        print(f"  {scale_name}: Neutral={n_neutral} ({100*n_neutral/len(scale_labels):.1f}%), "
              f"Long={n_long} ({100*n_long/len(scale_labels):.1f}%), "
              f"Short={n_short} ({100*n_short/len(scale_labels):.1f}%)")

    # Save time context and labels
    time_context = torch.tensor(
        np.stack([all_valid_sin, all_valid_cos], axis=1),
        dtype=torch.float32
    )
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    time_to_close_tensor = torch.tensor(all_valid_ttc, dtype=torch.float32)

    torch.save(time_context, embeddings_dir / "time_context.pt")
    torch.save(labels_tensor, embeddings_dir / "labels.pt")
    torch.save(time_to_close_tensor, embeddings_dir / "time_to_close.pt")

    # Save multi-scale labels
    for scale_name, scale_labels in multiscale_labels.items():
        torch.save(
            torch.tensor(scale_labels, dtype=torch.long),
            embeddings_dir / f"labels_{scale_name}.pt"
        )

    print(f"\nSaved time_context: {time_context.shape}")
    print(f"Saved labels: {labels_tensor.shape}")
    print(f"Saved multi-scale labels: labels_short.pt, labels_medium.pt, labels_long.pt")

    # Process each JEPA model sequentially to save VRAM
    jepa_configs = [
        ("short", args.short_jepa_checkpoint, short_context_len),
        ("med", args.med_jepa_checkpoint, med_context_len),
        ("long", args.long_jepa_checkpoint, long_context_len),
    ]

    for scale_name, ckpt_path, context_len in jepa_configs:
        print(f"\n{'='*50}")
        print(f"Processing {scale_name} JEPA (context_len={context_len})")
        print("=" * 50)

        # Load model
        model = LeJEPA.load(ckpt_path, map_location=str(device))
        model = model.to(device)
        model.eval()

        embeddings = []

        with torch.no_grad():
            for day_offset, day_indices, df in tqdm(valid_indices_per_day, desc=f"  {scale_name} embeddings"):
                # Prepare data tensor for this day (only numeric columns)
                numeric_df = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32', 'float', 'int'])
                data_tensor = torch.tensor(numeric_df.values, dtype=torch.float32)
                data_tensor = torch.nan_to_num(data_tensor, nan=0.0, posinf=10.0, neginf=-10.0)
                data_tensor = torch.clamp(data_tensor, -10, 10)

                # Process in batches
                batch_patches = []
                for i in day_indices:
                    start_idx = i - context_len
                    if start_idx >= 0:
                        patch = data_tensor[start_idx:i]
                    else:
                        # Pad with zeros if not enough history
                        patch = torch.zeros(context_len, data_tensor.shape[1])
                        available = data_tensor[:i]
                        patch[-len(available):] = available
                    batch_patches.append(patch)

                if batch_patches:
                    batch = torch.stack(batch_patches).to(device)

                    # Process in smaller batches to avoid OOM
                    batch_size = 512
                    for batch_start in range(0, len(batch), batch_size):
                        batch_end = min(batch_start + batch_size, len(batch))
                        mini_batch = batch[batch_start:batch_end]

                        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                            emb = model.encode(mini_batch)
                        embeddings.append(emb.float().cpu())

        # Concatenate and save
        all_embeddings = torch.cat(embeddings, dim=0)
        output_path = embeddings_dir / f"{scale_name}_embeddings.pt"
        torch.save(all_embeddings, output_path)
        print(f"  Saved {scale_name}_embeddings: {all_embeddings.shape}")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Save metadata
    metadata = {
        "total_samples": total_samples,
        "short_context_len": short_context_len,
        "med_context_len": med_context_len,
        "long_context_len": long_context_len,
        "lookahead": lookahead,
        "threshold_pct": args.threshold_pct,
        "num_files": len(files),
        "created_at": datetime.now().isoformat(),
    }
    with open(embeddings_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nEmbeddings saved to: {embeddings_dir}")
    return embeddings_dir


# =============================================================================
# Dataset with Scheduled Masking
# =============================================================================


class MaskedPolicyDataset(Dataset):
    """
    Dataset that loads pre-computed embeddings with scheduled masking.

    Applies random masking during training to simulate morning trading
    when not all scales have enough history.
    """

    def __init__(
        self,
        embeddings_dir: Path,
        mask_schedule: str = "realistic",
        apply_masking: bool = True,
        binary_mode: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            embeddings_dir: Path to pre-computed embeddings
            mask_schedule: "realistic" (time-based) or "uniform" (random)
            apply_masking: Whether to apply masking (False for validation)
            binary_mode: If True, filter out Neutral and do Long(0) vs Short(1)
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.mask_schedule = mask_schedule
        self.apply_masking = apply_masking
        self.binary_mode = binary_mode

        # Load all embeddings
        print(f"Loading embeddings from {embeddings_dir}...")
        short_emb = torch.load(embeddings_dir / "short_embeddings.pt")
        med_emb = torch.load(embeddings_dir / "med_embeddings.pt")
        long_emb = torch.load(embeddings_dir / "long_embeddings.pt")
        time_context = torch.load(embeddings_dir / "time_context.pt")
        labels = torch.load(embeddings_dir / "labels.pt")
        time_to_close = torch.load(embeddings_dir / "time_to_close.pt")

        # Load metadata
        with open(embeddings_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        if binary_mode:
            # Filter out Neutral (label 0), keep only Long (1) and Short (2)
            # Remap: Long (1) -> 0, Short (2) -> 1
            mask = labels != 0  # Keep Long and Short
            self.short_emb = short_emb[mask]
            self.med_emb = med_emb[mask]
            self.long_emb = long_emb[mask]
            self.time_context = time_context[mask]
            self.time_to_close = time_to_close[mask]
            # Remap labels: Long (1) -> 0, Short (2) -> 1
            self.labels = (labels[mask] - 1).long()
            print(f"  Binary mode: filtered to {len(self.labels)} samples (Long={int((self.labels==0).sum())}, Short={int((self.labels==1).sum())})")
        else:
            self.short_emb = short_emb
            self.med_emb = med_emb
            self.long_emb = long_emb
            self.time_context = time_context
            self.labels = labels
            self.time_to_close = time_to_close

        print(f"  Loaded {len(self.labels)} samples")
        print(f"  Embedding dim: {self.short_emb.shape[1]}")

    def __len__(self) -> int:
        return len(self.labels)

    def _get_mask(self, idx: int) -> torch.Tensor:
        """
        Generate mask based on schedule.

        Returns:
            mask: [3] tensor with 1 for available, 0 for missing
        """
        if not self.apply_masking:
            # During validation, use all available (full mask)
            return torch.ones(3)

        if self.mask_schedule == "realistic":
            # Use time_to_close to determine realistic mask
            # time_to_close: 1.0 at open, 0.0 at close
            ttc = self.time_to_close[idx].item()
            time_elapsed = 1.0 - ttc

            # Short: always available after market open
            # Medium: available after ~45 min (11.5% of day)
            # Long: available after ~90 min (23% of day)
            mask = torch.zeros(3)
            mask[0] = 1.0  # Short always available
            mask[1] = float(time_elapsed >= 0.115)  # ~45 min
            mask[2] = float(time_elapsed >= 0.231)  # ~90 min
            return mask

        elif self.mask_schedule == "uniform":
            # Random masking for data augmentation
            r = random.random()
            if r < 0.15:
                return torch.tensor([1.0, 0.0, 0.0])  # Morning: short only
            elif r < 0.45:
                return torch.tensor([1.0, 1.0, 0.0])  # Mid-morning: short + med
            else:
                return torch.tensor([1.0, 1.0, 1.0])  # Afternoon: all

        else:
            return torch.ones(3)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample with mask applied.

        Returns:
            emb_short, emb_med, emb_long, time_context, mask, label
        """
        mask = self._get_mask(idx)

        # Zero out embeddings for masked scales
        emb_short = self.short_emb[idx]
        emb_med = self.med_emb[idx] * mask[1]
        emb_long = self.long_emb[idx] * mask[2]

        return (
            emb_short,
            emb_med,
            emb_long,
            self.time_context[idx],
            mask,
            self.labels[idx],
        )

    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of labels."""
        unique, counts = np.unique(self.labels.numpy(), return_counts=True)
        return {
            "NEUTRAL": int(counts[unique == 0][0]) if 0 in unique else 0,
            "LONG": int(counts[unique == 1][0]) if 1 in unique else 0,
            "SHORT": int(counts[unique == 2][0]) if 2 in unique else 0,
        }


class MultiScaleDataset(Dataset):
    """
    Dataset for MultiScalePolicy training with scale-specific MAE labels.

    Unlike MaskedPolicyDataset which uses a single label, this dataset
    loads three sets of labels (short/medium/long) so each classifier
    head can be trained on scale-appropriate targets.
    """

    def __init__(
        self,
        embeddings_dir: Path,
        mask_schedule: str = "realistic",
        apply_masking: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            embeddings_dir: Path to pre-computed embeddings
            mask_schedule: "realistic" (time-based) or "uniform" (random)
            apply_masking: Whether to apply masking (False for validation)
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.mask_schedule = mask_schedule
        self.apply_masking = apply_masking

        # Load embeddings
        print(f"Loading multi-scale embeddings from {embeddings_dir}...")
        self.short_emb = torch.load(embeddings_dir / "short_embeddings.pt")
        self.med_emb = torch.load(embeddings_dir / "med_embeddings.pt")
        self.long_emb = torch.load(embeddings_dir / "long_embeddings.pt")
        self.time_context = torch.load(embeddings_dir / "time_context.pt")
        self.time_to_close = torch.load(embeddings_dir / "time_to_close.pt")

        # Load multi-scale labels
        labels_short_path = embeddings_dir / "labels_short.pt"
        labels_med_path = embeddings_dir / "labels_medium.pt"
        labels_long_path = embeddings_dir / "labels_long.pt"

        if not labels_short_path.exists():
            raise FileNotFoundError(
                f"Multi-scale labels not found at {labels_short_path}. "
                "Run with --precompute-only to generate MAE labels first."
            )

        self.labels_short = torch.load(labels_short_path)
        self.labels_med = torch.load(labels_med_path)
        self.labels_long = torch.load(labels_long_path)

        # Load metadata
        with open(embeddings_dir / "metadata.json", "r") as f:
            self.metadata = json.load(f)

        print(f"  Loaded {len(self.labels_short)} samples")
        print(f"  Embedding dim: {self.short_emb.shape[1]}")

        # Print label distributions
        for name, labels in [
            ("Short", self.labels_short),
            ("Medium", self.labels_med),
            ("Long", self.labels_long),
        ]:
            n_neutral = (labels == 0).sum().item()
            n_long = (labels == 1).sum().item()
            n_short = (labels == 2).sum().item()
            total = len(labels)
            print(f"  {name} labels: N={n_neutral} ({100*n_neutral/total:.1f}%), "
                  f"L={n_long} ({100*n_long/total:.1f}%), "
                  f"S={n_short} ({100*n_short/total:.1f}%)")

    def __len__(self) -> int:
        return len(self.labels_short)

    def _get_mask(self, idx: int) -> torch.Tensor:
        """Generate mask based on schedule."""
        if not self.apply_masking:
            return torch.ones(3)

        if self.mask_schedule == "realistic":
            ttc = self.time_to_close[idx].item()
            time_elapsed = 1.0 - ttc
            mask = torch.zeros(3)
            mask[0] = 1.0
            mask[1] = float(time_elapsed >= 0.115)
            mask[2] = float(time_elapsed >= 0.231)
            return mask
        elif self.mask_schedule == "uniform":
            r = random.random()
            if r < 0.15:
                return torch.tensor([1.0, 0.0, 0.0])
            elif r < 0.45:
                return torch.tensor([1.0, 1.0, 0.0])
            else:
                return torch.tensor([1.0, 1.0, 1.0])
        else:
            return torch.ones(3)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample with multi-scale labels.

        Returns:
            emb_short, emb_med, emb_long, time_context, mask,
            label_short, label_med, label_long
        """
        mask = self._get_mask(idx)

        emb_short = self.short_emb[idx]
        emb_med = self.med_emb[idx] * mask[1]
        emb_long = self.long_emb[idx] * mask[2]

        return (
            emb_short,
            emb_med,
            emb_long,
            self.time_context[idx],
            mask,
            self.labels_short[idx],
            self.labels_med[idx],
            self.labels_long[idx],
        )


# =============================================================================
# Training Functions
# =============================================================================


def train_masked_policy_epoch(
    model: MaskedGatedPolicy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_every: int = 100,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    attn_entropy_sum = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        emb_short, emb_med, emb_long, time_ctx, mask, labels = batch

        emb_short = emb_short.to(device)
        emb_med = emb_med.to(device)
        emb_long = emb_long.to(device)
        time_ctx = time_ctx.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, attn_weights = model(emb_short, emb_med, emb_long, time_ctx, mask)

        loss = criterion(logits, labels)

        # Optional: entropy regularization on attention weights
        # Encourages usage of all available heads
        attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=1).mean()
        attn_entropy_sum += attn_entropy.item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % log_every == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct/total:.3f}",
            })

    return {
        "train_loss": total_loss / len(dataloader),
        "train_acc": correct / total,
        "attn_entropy": attn_entropy_sum / len(dataloader),
    }


def validate_masked_policy(
    model: MaskedGatedPolicy,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    binary_mode: bool = False,
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # Per-class metrics
    n_classes = 2 if binary_mode else 3
    class_correct = {i: 0 for i in range(n_classes)}
    class_total = {i: 0 for i in range(n_classes)}
    class_pred = {i: 0 for i in range(n_classes)}

    # Attention weight analysis
    all_attn_weights = []

    with torch.no_grad():
        for batch in dataloader:
            emb_short, emb_med, emb_long, time_ctx, mask, labels = batch

            emb_short = emb_short.to(device)
            emb_med = emb_med.to(device)
            emb_long = emb_long.to(device)
            time_ctx = time_ctx.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            logits, attn_weights = model(emb_short, emb_med, emb_long, time_ctx, mask)
            all_attn_weights.append(attn_weights.cpu())

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Per-class tracking
            for i in range(n_classes):
                class_mask = labels == i
                class_total[i] += class_mask.sum().item()
                class_correct[i] += ((preds == labels) & class_mask).sum().item()
                class_pred[i] += (preds == i).sum().item()

    # Compute per-class metrics
    metrics = {
        "val_loss": total_loss / len(dataloader),
        "val_acc": correct / total,
    }

    if binary_mode:
        # Binary mode: Long(0), Short(1)
        class_names = ["long", "short"]
    else:
        class_names = ["neutral", "long", "short"]

    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            metrics[f"{name}_recall"] = class_correct[i] / class_total[i]
        else:
            metrics[f"{name}_recall"] = 0.0

        if class_pred[i] > 0:
            metrics[f"{name}_precision"] = class_correct[i] / class_pred[i]
        else:
            metrics[f"{name}_precision"] = 0.0

    # Attention weight statistics
    all_attn = torch.cat(all_attn_weights, dim=0)
    metrics["attn_short_mean"] = all_attn[:, 0].mean().item()
    metrics["attn_med_mean"] = all_attn[:, 1].mean().item()
    metrics["attn_long_mean"] = all_attn[:, 2].mean().item()

    return metrics


# =============================================================================
# MultiScalePolicy Training Functions
# =============================================================================


def train_multiscale_epoch(
    model: MultiScalePolicy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    head_weight: float = 1.0,
    combined_weight: float = 0.5,
    class_weights: Optional[torch.Tensor] = None,
    log_every: int = 100,
) -> Dict[str, float]:
    """Train MultiScalePolicy for one epoch."""
    model.train()

    total_loss = 0.0
    loss_components = {"short": 0.0, "med": 0.0, "long": 0.0, "combined": 0.0}
    correct = {"short": 0, "med": 0, "long": 0, "combined": 0}
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        emb_short, emb_med, emb_long, time_ctx, mask, lbl_short, lbl_med, lbl_long = batch

        emb_short = emb_short.to(device)
        emb_med = emb_med.to(device)
        emb_long = emb_long.to(device)
        time_ctx = time_ctx.to(device)
        mask = mask.to(device)
        lbl_short = lbl_short.to(device)
        lbl_med = lbl_med.to(device)
        lbl_long = lbl_long.to(device)

        optimizer.zero_grad()

        # Compute loss with individual head losses
        loss, loss_dict = model.compute_loss(
            emb_short, emb_med, emb_long, time_ctx,
            lbl_short, lbl_med, lbl_long,
            mask=mask,
            class_weights=class_weights,
            head_weight=head_weight,
            combined_weight=combined_weight,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loss_components["short"] += loss_dict["loss_short"].item()
        loss_components["med"] += loss_dict["loss_med"].item()
        loss_components["long"] += loss_dict["loss_long"].item()
        loss_components["combined"] += loss_dict["loss_combined"].item()

        # Get predictions for accuracy tracking
        with torch.no_grad():
            combined_logits, gate_weights, head_logits = model.forward(
                emb_short, emb_med, emb_long, time_ctx, mask, return_head_logits=True
            )

            correct["short"] += (head_logits["short"].argmax(1) == lbl_short).sum().item()
            correct["med"] += (head_logits["med"].argmax(1) == lbl_med).sum().item()
            correct["long"] += (head_logits["long"].argmax(1) == lbl_long).sum().item()
            correct["combined"] += (combined_logits.argmax(1) == lbl_med).sum().item()

        total += lbl_short.size(0)

        if batch_idx % log_every == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc_s": f"{correct['short']/total:.3f}",
                "acc_m": f"{correct['med']/total:.3f}",
                "acc_l": f"{correct['long']/total:.3f}",
            })

    n_batches = len(dataloader)
    return {
        "train_loss": total_loss / n_batches,
        "train_loss_short": loss_components["short"] / n_batches,
        "train_loss_med": loss_components["med"] / n_batches,
        "train_loss_long": loss_components["long"] / n_batches,
        "train_loss_combined": loss_components["combined"] / n_batches,
        "train_acc_short": correct["short"] / total,
        "train_acc_med": correct["med"] / total,
        "train_acc_long": correct["long"] / total,
        "train_acc_combined": correct["combined"] / total,
    }


def validate_multiscale(
    model: MultiScalePolicy,
    dataloader: DataLoader,
    device: torch.device,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Validate MultiScalePolicy."""
    model.eval()

    total_loss = 0.0
    loss_components = {"short": 0.0, "med": 0.0, "long": 0.0, "combined": 0.0}
    correct = {"short": 0, "med": 0, "long": 0, "combined": 0}
    total = 0

    # Per-class metrics for each head
    class_correct = {
        "short": {0: 0, 1: 0, 2: 0},
        "med": {0: 0, 1: 0, 2: 0},
        "long": {0: 0, 1: 0, 2: 0},
    }
    class_total = {
        "short": {0: 0, 1: 0, 2: 0},
        "med": {0: 0, 1: 0, 2: 0},
        "long": {0: 0, 1: 0, 2: 0},
    }

    all_gate_weights = []

    with torch.no_grad():
        for batch in dataloader:
            emb_short, emb_med, emb_long, time_ctx, mask, lbl_short, lbl_med, lbl_long = batch

            emb_short = emb_short.to(device)
            emb_med = emb_med.to(device)
            emb_long = emb_long.to(device)
            time_ctx = time_ctx.to(device)
            mask = mask.to(device)
            lbl_short = lbl_short.to(device)
            lbl_med = lbl_med.to(device)
            lbl_long = lbl_long.to(device)

            loss, loss_dict = model.compute_loss(
                emb_short, emb_med, emb_long, time_ctx,
                lbl_short, lbl_med, lbl_long,
                mask=mask,
                class_weights=class_weights,
            )

            total_loss += loss.item()
            loss_components["short"] += loss_dict["loss_short"].item()
            loss_components["med"] += loss_dict["loss_med"].item()
            loss_components["long"] += loss_dict["loss_long"].item()
            loss_components["combined"] += loss_dict["loss_combined"].item()

            # Get predictions
            combined_logits, gate_weights, head_logits = model.forward(
                emb_short, emb_med, emb_long, time_ctx, mask, return_head_logits=True
            )
            all_gate_weights.append(gate_weights.cpu())

            pred_short = head_logits["short"].argmax(1)
            pred_med = head_logits["med"].argmax(1)
            pred_long = head_logits["long"].argmax(1)
            pred_combined = combined_logits.argmax(1)

            correct["short"] += (pred_short == lbl_short).sum().item()
            correct["med"] += (pred_med == lbl_med).sum().item()
            correct["long"] += (pred_long == lbl_long).sum().item()
            correct["combined"] += (pred_combined == lbl_med).sum().item()

            total += lbl_short.size(0)

            # Per-class tracking
            for head_name, preds, labels in [
                ("short", pred_short, lbl_short),
                ("med", pred_med, lbl_med),
                ("long", pred_long, lbl_long),
            ]:
                for c in [0, 1, 2]:
                    mask_c = labels == c
                    class_total[head_name][c] += mask_c.sum().item()
                    class_correct[head_name][c] += ((preds == labels) & mask_c).sum().item()

    n_batches = len(dataloader)
    metrics = {
        "val_loss": total_loss / n_batches,
        "val_loss_short": loss_components["short"] / n_batches,
        "val_loss_med": loss_components["med"] / n_batches,
        "val_loss_long": loss_components["long"] / n_batches,
        "val_loss_combined": loss_components["combined"] / n_batches,
        "val_acc_short": correct["short"] / total,
        "val_acc_med": correct["med"] / total,
        "val_acc_long": correct["long"] / total,
        "val_acc_combined": correct["combined"] / total,
    }

    # Per-class recall for each head
    class_names = ["neutral", "long", "short"]
    for head_name in ["short", "med", "long"]:
        for c, cname in enumerate(class_names):
            if class_total[head_name][c] > 0:
                metrics[f"{head_name}_{cname}_recall"] = class_correct[head_name][c] / class_total[head_name][c]
            else:
                metrics[f"{head_name}_{cname}_recall"] = 0.0

    # Gate weight statistics
    all_gates = torch.cat(all_gate_weights, dim=0)
    metrics["gate_short_mean"] = all_gates[:, 0].mean().item()
    metrics["gate_med_mean"] = all_gates[:, 1].mean().item()
    metrics["gate_long_mean"] = all_gates[:, 2].mean().item()

    return metrics


# =============================================================================
# Main Training Loop
# =============================================================================


def train_masked_policy(args: argparse.Namespace) -> None:
    """Main training function for MaskedGatedPolicy."""
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Check if we need to precompute embeddings
    embeddings_dir = Path(args.embeddings_dir)
    if args.precompute_only or not (embeddings_dir / "metadata.json").exists():
        if not all([args.short_jepa_checkpoint, args.med_jepa_checkpoint, args.long_jepa_checkpoint]):
            raise ValueError(
                "Must provide --short-jepa-checkpoint, --med-jepa-checkpoint, "
                "and --long-jepa-checkpoint for embedding pre-computation"
            )
        precompute_embeddings(args, device)

        if args.precompute_only:
            print("\nPre-computation complete. Exiting.")
            return

    # Load datasets
    print("\n" + "=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)

    binary_mode = getattr(args, 'binary_mode', False)
    full_dataset = MaskedPolicyDataset(
        embeddings_dir,
        mask_schedule=args.mask_schedule,
        apply_masking=True,
        binary_mode=binary_mode,
    )

    # Split into train/val
    n_total = len(full_dataset)
    n_train = int(n_total * args.train_split)
    n_val = n_total - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )

    # Create validation dataset without masking
    val_dataset_no_mask = MaskedPolicyDataset(
        embeddings_dir,
        mask_schedule="none",
        apply_masking=False,
        binary_mode=binary_mode,
    )
    val_indices = val_dataset.indices
    val_dataset_no_mask = torch.utils.data.Subset(val_dataset_no_mask, val_indices)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset_no_mask,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Get embedding dimension from data
    sample = full_dataset[0]
    emb_dim = sample[0].shape[0]

    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    n_positions = 2 if binary_mode else 3
    if args.direct_policy:
        print("Using DirectPolicy (simple MLP, no attention)")
        model = DirectPolicy(
            emb_dim=emb_dim,
            time_dim=2,  # Sin/cos time encoding
            hidden_dim=args.hidden_dim,
            n_positions=n_positions,
            n_scales=3,
            dropout=args.dropout,
        )
    else:
        print("Using MaskedGatedPolicy (with attention)")
        model = MaskedGatedPolicy(
            emb_dim=emb_dim,
            time_dim=2,
            hidden_dim=args.hidden_dim,
            n_positions=n_positions,
            n_scales=3,
            dropout=args.dropout,
        )
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    class_weights = None
    if args.class_weights:
        if args.class_weights == "auto":
            # Compute inverse frequency weights
            labels = full_dataset.labels.numpy()
            n_classes = 2 if binary_mode else 3
            counts = np.bincount(labels, minlength=n_classes)
            # Inverse frequency, normalized
            weights = len(labels) / (n_classes * counts + 1e-6)
            class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
            print(f"Auto class weights: {class_weights.tolist()}")
        else:
            # Parse comma-separated weights
            weights = [float(w) for w in args.class_weights.split(",")]
            class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
            print(f"Manual class weights: {class_weights.tolist()}")

    if args.use_focal_loss:
        criterion = FocalLoss(gamma=args.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Checkpoint management
    project_root = Path(__file__).parent.parent
    mgr = CheckpointManager(project_root / "checkpoints", project_root)

    excluded_keys = {"model_type", "checkpoint_id", "precompute_only"}
    config = {k: v for k, v in vars(args).items() if k not in excluded_keys}
    entry = mgr.get_or_create(
        args.checkpoint_id,
        model_type="masked-gated-policy",
        config=config,
    )
    checkpoint_dir = Path(entry["path"])
    print(f"\nCheckpoint directory: {checkpoint_dir}")

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    best_val_acc = 0.0
    training_history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 50)

        train_metrics = train_masked_policy_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.log_every
        )

        val_metrics = validate_masked_policy(
            model, val_loader, criterion, device, binary_mode=binary_mode
        )

        scheduler.step()

        # Log metrics
        print(f"\nTrain Loss: {train_metrics['train_loss']:.4f}, Acc: {train_metrics['train_acc']:.3f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']:.3f}")
        if binary_mode:
            print(f"  Long  - P: {val_metrics.get('long_precision', 0):.3f}, R: {val_metrics.get('long_recall', 0):.3f}")
            print(f"  Short - P: {val_metrics.get('short_precision', 0):.3f}, R: {val_metrics.get('short_recall', 0):.3f}")
        else:
            print(f"  Neutral - P: {val_metrics['neutral_precision']:.3f}, R: {val_metrics['neutral_recall']:.3f}")
            print(f"  Long    - P: {val_metrics['long_precision']:.3f}, R: {val_metrics['long_recall']:.3f}")
            print(f"  Short   - P: {val_metrics['short_precision']:.3f}, R: {val_metrics['short_recall']:.3f}")
        print(f"  Attn weights (mean): S={val_metrics['attn_short_mean']:.3f}, "
              f"M={val_metrics['attn_med_mean']:.3f}, L={val_metrics['attn_long_mean']:.3f}")

        # Save history
        epoch_record = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        training_history.append(epoch_record)

        # Save best model
        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            best_path = checkpoint_dir / "policy_best.pt"
            model.save(str(best_path))
            print(f"  -> New best model saved (acc: {best_val_acc:.3f})")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = checkpoint_dir / f"policy_epoch_{epoch:04d}.pt"
            model.save(str(ckpt_path))

        # Save training history
        save_training_history(training_history, checkpoint_dir / "training_history.json")
        generate_training_charts(training_history, checkpoint_dir / "training_charts.html")

    # Save final model
    final_path = checkpoint_dir / "policy_final.pt"
    model.save(str(final_path))

    print("\n" + "=" * 70)
    print(f"Training complete! Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 70)


def train_multiscale_policy(args: argparse.Namespace) -> None:
    """Main training function for MultiScalePolicy with MAE labels."""
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Check if we need to precompute embeddings
    embeddings_dir = Path(args.embeddings_dir)
    if args.precompute_only or not (embeddings_dir / "metadata.json").exists():
        if not all([args.short_jepa_checkpoint, args.med_jepa_checkpoint, args.long_jepa_checkpoint]):
            raise ValueError(
                "Must provide --short-jepa-checkpoint, --med-jepa-checkpoint, "
                "and --long-jepa-checkpoint for embedding pre-computation"
            )
        precompute_embeddings(args, device)

        if args.precompute_only:
            print("\nPre-computation complete. Exiting.")
            return

    # Check for multi-scale labels
    if not (embeddings_dir / "labels_short.pt").exists():
        raise FileNotFoundError(
            f"Multi-scale labels not found. Run with --precompute-only first to generate MAE labels."
        )

    # Load datasets
    print("\n" + "=" * 70)
    print("LOADING MULTI-SCALE DATASETS")
    print("=" * 70)

    full_dataset = MultiScaleDataset(
        embeddings_dir,
        mask_schedule=args.mask_schedule,
        apply_masking=True,
    )

    # Split into train/val
    n_total = len(full_dataset)
    n_train = int(n_total * args.train_split)
    n_val = n_total - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val]
    )

    # Create validation dataset without masking
    val_dataset_no_mask = MultiScaleDataset(
        embeddings_dir,
        mask_schedule="none",
        apply_masking=False,
    )
    val_indices = val_dataset.indices
    val_dataset_no_mask = torch.utils.data.Subset(val_dataset_no_mask, val_indices)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset_no_mask,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Get embedding dimension from data
    sample = full_dataset[0]
    emb_dim = sample[0].shape[0]

    # Create model
    print("\n" + "=" * 70)
    print("CREATING MULTISCALE POLICY MODEL")
    print("=" * 70)

    model = MultiScalePolicy(
        emb_dim=emb_dim,
        time_dim=2,
        hidden_dim=args.hidden_dim,
        n_positions=3,  # Always 3 for multi-scale (Neutral/Long/Short)
        dropout=args.dropout,
    )
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Checkpoint management
    project_root = Path(__file__).parent.parent
    mgr = CheckpointManager(project_root / "checkpoints", project_root)

    excluded_keys = {"model_type", "checkpoint_id", "precompute_only"}
    config = {k: v for k, v in vars(args).items() if k not in excluded_keys}
    entry = mgr.get_or_create(
        args.checkpoint_id,
        model_type="multi-scale-policy",
        config=config,
    )
    checkpoint_dir = Path(entry["path"])
    print(f"\nCheckpoint directory: {checkpoint_dir}")

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING MULTI-SCALE TRAINING")
    print("=" * 70)

    best_val_acc = 0.0
    training_history = []

    head_weight = getattr(args, 'head_weight', 1.0)
    combined_weight = getattr(args, 'combined_weight', 0.5)
    print(f"Loss weights: head_weight={head_weight}, combined_weight={combined_weight}")

    # Compute class weights based on label distribution (use medium labels as reference)
    labels_med = full_dataset.labels_med.numpy()
    class_counts = np.bincount(labels_med, minlength=3)
    # Inverse frequency weighting: more weight to minority classes
    class_weights = len(labels_med) / (3 * class_counts + 1e-6)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights (N/L/S): {class_weights.tolist()}")

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print("=" * 50)

        train_metrics = train_multiscale_epoch(
            model, train_loader, optimizer, device, epoch,
            head_weight=head_weight,
            combined_weight=combined_weight,
            class_weights=class_weights,
            log_every=args.log_every,
        )

        val_metrics = validate_multiscale(model, val_loader, device, class_weights=class_weights)

        scheduler.step()

        # Log metrics
        print(f"\nTrain Loss: {train_metrics['train_loss']:.4f}")
        print(f"  Acc - Short: {train_metrics['train_acc_short']:.3f}, "
              f"Med: {train_metrics['train_acc_med']:.3f}, "
              f"Long: {train_metrics['train_acc_long']:.3f}, "
              f"Combined: {train_metrics['train_acc_combined']:.3f}")

        print(f"\nVal Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Acc - Short: {val_metrics['val_acc_short']:.3f}, "
              f"Med: {val_metrics['val_acc_med']:.3f}, "
              f"Long: {val_metrics['val_acc_long']:.3f}, "
              f"Combined: {val_metrics['val_acc_combined']:.3f}")

        print(f"\n  Short head recall - N: {val_metrics['short_neutral_recall']:.3f}, "
              f"L: {val_metrics['short_long_recall']:.3f}, S: {val_metrics['short_short_recall']:.3f}")
        print(f"  Med head recall   - N: {val_metrics['med_neutral_recall']:.3f}, "
              f"L: {val_metrics['med_long_recall']:.3f}, S: {val_metrics['med_short_recall']:.3f}")
        print(f"  Long head recall  - N: {val_metrics['long_neutral_recall']:.3f}, "
              f"L: {val_metrics['long_long_recall']:.3f}, S: {val_metrics['long_short_recall']:.3f}")

        print(f"\n  Gate weights (mean): S={val_metrics['gate_short_mean']:.3f}, "
              f"M={val_metrics['gate_med_mean']:.3f}, L={val_metrics['gate_long_mean']:.3f}")

        # Save history
        epoch_record = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        training_history.append(epoch_record)

        # Use medium head accuracy as the main metric for best model selection
        # (since medium horizon is our primary trading target)
        val_acc = val_metrics["val_acc_med"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = checkpoint_dir / "policy_best.pt"
            model.save(str(best_path))
            print(f"  -> New best model saved (med_acc: {best_val_acc:.3f})")

        # Periodic checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = checkpoint_dir / f"policy_epoch_{epoch:04d}.pt"
            model.save(str(ckpt_path))

        # Save training history
        save_training_history(training_history, checkpoint_dir / "training_history.json")
        generate_training_charts(training_history, checkpoint_dir / "training_charts.html")

    # Save final model
    final_path = checkpoint_dir / "policy_final.pt"
    model.save(str(final_path))

    print("\n" + "=" * 70)
    print(f"Training complete! Best val accuracy (medium): {best_val_acc:.3f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 70)
