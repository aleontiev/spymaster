"""
LeJEPA training module.

Contains:
- JEPASlidingWindowDataset for JEPA training
- JEPA training functions
- JEPA-specific argument handling
"""

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.processing import create_dataloader
from src.data.synthetic import create_test_dataset
from src.model.lejepa import LeJEPA
from src.managers.checkpoint_manager import CheckpointManager

from train.common import set_seed, get_device, warmup_cosine_schedule
from train.visualization import generate_training_charts, save_training_history


def save_training_params(
    args: argparse.Namespace,
    feature_cols: List[str],
    output_path: Path,
    data_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save all training parameters to a JSON file.

    Args:
        args: Parsed command-line arguments (includes both explicit and default values)
        feature_cols: List of feature column names used in training
        output_path: Path to save the JSON file
        data_info: Optional dict with additional data info (date range, num_days, etc.)
    """
    # Convert args namespace to dict, handling non-serializable types
    params = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            params[key] = str(value)
        elif isinstance(value, (date, datetime)):
            params[key] = value.isoformat()
        elif value is None or isinstance(value, (str, int, float, bool, list)):
            params[key] = value
        else:
            params[key] = str(value)

    # Add features list
    params["features"] = feature_cols
    params["num_features"] = len(feature_cols)

    # Add data info if provided
    if data_info:
        params["data_info"] = data_info

    # Add metadata
    params["saved_at"] = datetime.now().isoformat()

    # Save to file
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Training parameters saved to: {output_path}")


# =============================================================================
# JEPA Datasets
# =============================================================================


class JEPASlidingWindowDataset(Dataset):
    """
    Sliding window dataset for JEPA training.

    Creates sliding windows from per-day data, returning (context, target) pairs.
    Context columns (T-3, T-2, T-1, PM, OR) are embedded in the feature columns.
    """

    def __init__(
        self,
        day_data_list: List[torch.Tensor],
        context_len: int,
        target_len: int,
        prediction_gap: int = 0,
        stride: int = 1,
    ):
        """
        Args:
            day_data_list: List of tensors, one per day [T_i, F]
            context_len: Number of candles for context window
            target_len: Number of candles for target window
            prediction_gap: Gap between context end and target start
            stride: Step size for sliding window
        """
        self.days = day_data_list
        self.context_len = context_len
        self.target_len = target_len
        self.prediction_gap = prediction_gap
        self.stride = stride

        # Build index mapping: global_idx → (day_idx, start_in_day)
        total_len = context_len + prediction_gap + target_len
        self.index_map = []
        for day_idx, day_data in enumerate(self.days):
            day_len = len(day_data)
            max_start = day_len - total_len
            if max_start >= 0:
                for start in range(0, max_start + 1, stride):
                    self.index_map.append((day_idx, start))

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        day_idx, start = self.index_map[idx]
        day_data = self.days[day_idx]

        context_end = start + self.context_len
        target_start = context_end + self.prediction_gap
        target_end = target_start + self.target_len

        context = day_data[start:context_end]
        target = day_data[target_start:target_end]

        return context, target


def create_sliding_window_dataset(
    df: pd.DataFrame,
    context_len: int,
    target_len: int,
    prediction_gap: int = 0,
    stride: int = 1,
) -> JEPASlidingWindowDataset:
    """
    Create a JEPA sliding window dataset from a DataFrame.

    Args:
        df: DataFrame with normalized features (excluding timestamp/date columns)
        context_len: Number of candles for context window
        target_len: Number of candles for target window
        prediction_gap: Gap between context end and target start
        stride: Step size for sliding window

    Returns:
        JEPASlidingWindowDataset ready for training
    """
    # Convert to tensor (float32 for training) and wrap as single-day list
    data = torch.tensor(df.values, dtype=torch.float32)

    return JEPASlidingWindowDataset(
        day_data_list=[data],  # Single "day" containing all data
        context_len=context_len,
        target_len=target_len,
        prediction_gap=prediction_gap,
        stride=stride,
        use_premarket=False,  # No premarket for synthetic/simple data
    )


# =============================================================================
# JEPA Training Functions
# =============================================================================


def create_jepa_optimizer(
    model: LeJEPA, lr: float, weight_decay: float
) -> Tuple[AdamW, list]:
    """Create AdamW optimizer with proper parameter groups for LeJEPA."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    return optimizer, param_groups


def train_jepa_epoch(
    model: LeJEPA,
    dataloader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
    global_step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    log_every: int = 10,
    use_amp: bool = True,
    accumulation_steps: int = 1,
) -> Tuple[Dict[str, float], int]:
    """Train LeJEPA for one epoch.

    Args:
        accumulation_steps: Number of gradient accumulation steps.
            Effective batch size = batch_size * accumulation_steps.
            Gradients are accumulated over this many batches before updating.
    """
    model.train()

    total_loss = 0.0
    total_pred_loss = 0.0
    total_sigreg_loss = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader):
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        context, target = batch
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Update learning rate based on global step
        lr = warmup_cosine_schedule(
            optimizer, global_step, warmup_steps, total_steps, base_lr
        )

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(x_context=context, x_target=target, return_loss=True)
                # Scale loss for gradient accumulation
                loss = output["loss"] / accumulation_steps
        else:
            output = model(x_context=context, x_target=target, return_loss=True)
            loss = output["loss"] / accumulation_steps

        loss.backward()

        # Track unscaled loss for logging
        total_loss += output["loss"].item()
        total_pred_loss += output["pred_loss"].item()
        total_sigreg_loss += output["sigreg_loss"].item()
        num_batches += 1

        # Update weights every accumulation_steps or at the end of epoch
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        if batch_idx % log_every == 0:
            print(
                f"  Epoch {epoch} | Step {batch_idx}/{len(dataloader)} | "
                f"Loss: {output['loss'].item():.4f} | Pred: {output['pred_loss'].item():.4f} | "
                f"SIGReg: {output['sigreg_loss'].item():.4f} | LR: {lr:.2e}"
            )

    metrics = {
        "loss": total_loss / num_batches,
        "pred_loss": total_pred_loss / num_batches,
        "sigreg_loss": total_sigreg_loss / num_batches,
    }

    return metrics, global_step


@torch.no_grad()
def validate_jepa(
    model: LeJEPA,
    dataloader: DataLoader,
    device: torch.device,
    embedding_dim: int,
    use_amp: bool = True,
) -> Dict[str, float]:
    """Validate LeJEPA model."""
    model.eval()

    total_pred_loss = 0.0
    num_batches = 0
    all_embeddings = []

    for batch in dataloader:
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        context, target = batch
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(x_context=context, x_target=target, return_loss=False)
        else:
            output = model(x_context=context, x_target=target, return_loss=False)

        pred_loss = torch.nn.functional.mse_loss(
            output["predicted_embedding"], output["target_embedding"]
        )
        total_pred_loss += pred_loss.item()
        num_batches += 1

        all_embeddings.append(output["context_embedding"].float().cpu())

    avg_pred_loss = total_pred_loss / num_batches
    metrics = {
        "val_loss": avg_pred_loss,
        "val_pred_loss": avg_pred_loss,
    }

    embeddings = torch.cat(all_embeddings, dim=0)
    metrics["embedding_std"] = embeddings.std().item()
    metrics["embedding_mean_norm"] = embeddings.norm(dim=1).mean().item()

    dim_variance = embeddings.var(dim=0)
    metrics["dim_variance_mean"] = dim_variance.mean().item()
    metrics["dim_variance_std"] = dim_variance.std().item()

    # Compute effective rank for collapse detection
    cov = torch.cov(embeddings.T)
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.sort(descending=True).values
    threshold = 0.01 * eigenvalues[0]
    effective_rank = (eigenvalues > threshold).sum().item()
    metrics["effective_rank"] = effective_rank
    metrics["top_eigenvalue"] = eigenvalues[0].item()

    return metrics


# =============================================================================
# JEPA Argument Handling
# =============================================================================


def add_jepa_args(parser: argparse.ArgumentParser) -> None:
    """Add JEPA-specific arguments to parser."""
    # LeJEPA model parameters
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=64,
        help="Embedding dimension (latent state size)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--ff-dim",
        type=int,
        default=2048,
        help="Feed-forward dimension",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability",
    )
    parser.add_argument(
        "--ema-momentum",
        type=float,
        default=0.90,
        help="EMA momentum for target encoder",
    )
    parser.add_argument(
        "--lambda-sigreg",
        type=float,
        default=0.5,
        help="Regularization loss weight",
    )
    parser.add_argument(
        "--reg-type",
        type=str,
        default="sigreg",
        choices=["vicreg", "sigreg"],
        help="Regularization type",
    )
    parser.add_argument(
        "--sigreg-threshold",
        type=float,
        default=60.0,
        help="SIGReg loss threshold for best model selection",
    )
    parser.add_argument(
        "--min-effective-rank-pct",
        type=float,
        default=0.05,
        help="Minimum effective rank as fraction of embedding_dim",
    )
    parser.add_argument(
        "--repr-warmup-epochs",
        type=int,
        default=2,
        help="Representation warmup epochs (SIGReg only)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (effective_batch = batch_size * steps)",
    )


# =============================================================================
# Main Training Function
# =============================================================================


def train_jepa(args: argparse.Namespace) -> None:
    """Main LeJEPA training function."""
    print("=" * 70)
    print("LeJEPA Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"\nDevice: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create dataset
    print("\n" + "=" * 70)
    print("Creating Dataset")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    normalized_dir = data_dir / "training-1m-normalized" / args.underlying

    print(f"Data directory: {data_dir}")
    print(f"Normalized data: {normalized_dir}")
    print(f"Underlying: {args.underlying}")
    if args.start_date:
        print(f"Start date: {args.start_date}")
    if args.end_date:
        print(f"End date: {args.end_date}")

    # Track features and dates for training_params.json
    feature_cols: List[str] = []
    dates: List[date] = []
    num_train_days: Optional[int] = None
    num_val_days: Optional[int] = None

    # Use normalized data (properly scaled, consistent features)
    if normalized_dir.exists():
        print("\n" + "-" * 70)
        print("Loading Normalized Training Data")
        print("-" * 70)

        # Parse date range
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

        # Load normalized data
        all_files = []
        for month_dir in sorted(normalized_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            for parquet_file in sorted(month_dir.glob("*.parquet")):
                try:
                    month_str = month_dir.name
                    day_str = parquet_file.stem
                    file_date = datetime.strptime(f"{month_str}-{day_str}", "%Y-%m-%d").date()
                    if start_date and file_date < start_date:
                        continue
                    if end_date and file_date > end_date:
                        continue
                    all_files.append((file_date, parquet_file))
                except ValueError:
                    continue

        all_files.sort(key=lambda x: x[0])
        if args.max_files:
            all_files = all_files[:args.max_files]

        # Detect features from most recent file (canonical feature set)
        # New format: 95 features (70 core + 25 context columns), no special premarket rows
        latest_df = pd.read_parquet(all_files[-1][1])

        # Exclude non-numeric and metadata columns
        exclude_cols = {'timestamp', 'date', 'datetime', 'index', 'symbol', 'underlying'}
        feature_cols = [
            col for col in latest_df.columns
            if col.lower() not in exclude_cols
            and latest_df[col].dtype in ['float64', 'float32', 'int64', 'int32']
        ]
        feature_cols = sorted(feature_cols)  # Consistent ordering

        print(f"Detected {len(feature_cols)} features from normalized data (from latest file)")
        print(f"Features: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Features: {feature_cols}")

        # Load data - simple loop, no premarket handling needed
        day_tensors = []
        dates = []

        for file_date, parquet_file in tqdm(all_files, desc="Loading normalized data", unit="day"):
            df = pd.read_parquet(parquet_file)

            # Validate all expected features are present
            missing_cols = set(feature_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"File {parquet_file} is missing {len(missing_cols)} expected columns:\n"
                    f"  {sorted(missing_cols)[:10]}{'...' if len(missing_cols) > 10 else ''}\n"
                    f"This file has {len(df.columns)} columns, but canonical set has {len(feature_cols)}.\n"
                    f"You may need to regenerate training data for older dates."
                )

            selected = df[feature_cols].fillna(0)

            tensor = torch.tensor(selected.values, dtype=torch.float32)
            day_tensors.append(tensor)
            dates.append(file_date)

        print(f"Loaded {len(dates)} days")
        if dates:
            print(f"Date range: {dates[0]} to {dates[-1]}")
            print(f"Features: {len(feature_cols)}")
            print(f"Candles per day: {day_tensors[0].shape[0]}")

        # Validate data quality
        all_data = torch.cat(day_tensors, dim=0)
        print(f"\nData validation:")
        for i, col in enumerate(feature_cols):
            vals = all_data[:, i]
            zeros_pct = (vals == 0).float().mean().item() * 100
            std = vals.std().item()
            if zeros_pct > 50 or std < 0.001:
                print(f"  Warning: {col}: {zeros_pct:.1f}% zeros, std={std:.4f}")

        # Split into train/val by date
        split_idx = int(len(dates) * args.train_split)
        train_tensors = day_tensors[:split_idx]
        val_tensors = day_tensors[split_idx:]
        num_train_days = len(train_tensors)
        num_val_days = len(val_tensors)

        print(f"\nTrain: {num_train_days} days")
        print(f"Val: {num_val_days} days")

        train_dataset = JEPASlidingWindowDataset(
            day_data_list=train_tensors,
            context_len=args.context_len,
            target_len=args.target_len,
            prediction_gap=args.prediction_gap,
            stride=args.stride,
        )
        val_dataset = JEPASlidingWindowDataset(
            day_data_list=val_tensors,
            context_len=args.context_len,
            target_len=args.target_len,
            prediction_gap=args.prediction_gap,
            stride=args.stride,
        )
    else:
        print("\nUsing synthetic data (no normalized data directory found)")
        df = create_test_dataset(num_days=30, bars_per_day=390, random_seed=args.seed)
        feature_cols = list(df.columns)  # Track features for synthetic data
        split_idx = int(len(df) * args.train_split)
        df_train = df.iloc[:split_idx]
        df_val = df.iloc[split_idx:]

        train_dataset = create_sliding_window_dataset(
            df_train,
            context_len=args.context_len,
            target_len=args.target_len,
            prediction_gap=args.prediction_gap,
            stride=args.stride,
        )
        val_dataset = create_sliding_window_dataset(
            df_val,
            context_len=args.context_len,
            target_len=args.target_len,
            prediction_gap=args.prediction_gap,
            stride=args.stride,
        )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # Initialize model
    print("\n" + "=" * 70)
    print("Initializing Model")
    print("=" * 70)

    sample_context, sample_target = train_dataset[0]
    feature_dim = sample_context.shape[-1]
    context_len = sample_context.shape[0]
    target_len = sample_target.shape[0]
    print(f"Context: {context_len} candles × {feature_dim} features")
    print(f"Target: {target_len} candles × {feature_dim} features")

    model = LeJEPA(
        input_dim=feature_dim,
        d_model=args.ff_dim // 4,
        nhead=args.num_heads,
        num_layers=args.num_layers,
        embedding_dim=args.embedding_dim,
        max_context_len=context_len,
        dropout=args.dropout,
        lambda_reg=args.lambda_sigreg,
        reg_type=args.reg_type,
    )
    print(f"Regularization: {args.reg_type.upper()}")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Calculate steps accounting for gradient accumulation
    accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)

    if args.compile and hasattr(torch, "compile"):
        # Use "default" mode with gradient accumulation since "reduce-overhead"
        # uses CUDA graphs which are incompatible with accumulation
        if accumulation_steps > 1:
            compile_mode = "default"
            print(f"Using torch.compile mode='default' (gradient accumulation incompatible with reduce-overhead)")
        else:
            compile_mode = "reduce-overhead"
        model.transformer = torch.compile(model.transformer, mode=compile_mode)
        model.predictor = torch.compile(model.predictor, mode=compile_mode)

    optimizer, _ = create_jepa_optimizer(model, args.lr, args.weight_decay)
    batches_per_epoch = len(train_loader)
    steps_per_epoch = (batches_per_epoch + accumulation_steps - 1) // accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    effective_batch_size = args.batch_size * accumulation_steps
    if accumulation_steps > 1:
        print(f"\nGradient Accumulation: {accumulation_steps} steps")
        print(f"  Micro-batch size: {args.batch_size}")
        print(f"  Effective batch size: {effective_batch_size}")
        print(f"  Batches per epoch: {batches_per_epoch}")
        print(f"  Weight updates per epoch: {steps_per_epoch}")

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0

    project_root = Path(__file__).parent.parent
    mgr = CheckpointManager(project_root / "checkpoints", project_root)

    if args.resume:
        resume_path = args.resume
        resume_entry = mgr.get(args.resume)
        if resume_entry:
            best_path = mgr.get_best_checkpoint(args.resume)
            if best_path:
                resume_path = str(best_path)
            else:
                resume_path = str(resume_entry.path / "lejepa_best.pt")

        # Find the latest epoch checkpoint (not best) for proper resumption
        resume_dir = Path(resume_path) if Path(resume_path).is_dir() else Path(resume_path).parent
        epoch_checkpoints = sorted(resume_dir.glob("lejepa_epoch_*.pt"))
        if epoch_checkpoints:
            resume_path = str(epoch_checkpoints[-1])  # Latest epoch

        print(f"\nResuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Checkpoint stores 1-indexed epoch, so this is the next epoch to train
        start_epoch = checkpoint.get("epoch", 0)
        global_step = checkpoint.get("global_step", 0)
        print(f"Resuming from epoch {start_epoch + 1}, step {global_step}")

    # Register checkpoint in manager
    excluded_keys = {"model_type", "checkpoint_id"}
    config = {k: v for k, v in vars(args).items() if k not in excluded_keys}
    entry = mgr.get_or_create(
        args.checkpoint_id,
        model_type="lejepa",
        config=config,
    )
    checkpoint_dir = Path(entry["path"])
    print(f"\nCheckpoint registered: {entry['name']} (ID: {entry['id'][:8]}...)")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Save training parameters (all args + features list)
    data_info = {
        "num_train_days": num_train_days,
        "num_val_days": num_val_days,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "date_range_start": str(dates[0]) if dates else None,
        "date_range_end": str(dates[-1]) if dates else None,
    }
    save_training_params(
        args=args,
        feature_cols=feature_cols,
        output_path=checkpoint_dir / "training_params.json",
        data_info=data_info,
    )

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    repr_warmup_epochs = getattr(args, 'repr_warmup_epochs', 0)
    original_lambda_reg = model.lambda_reg

    # Only show warmup message if we haven't already passed the warmup period
    if repr_warmup_epochs > 0 and start_epoch < repr_warmup_epochs:
        remaining_warmup = repr_warmup_epochs - start_epoch
        print(f"\n*** REPRESENTATION WARMUP: {remaining_warmup} epoch(s) remaining ***")
        print(f"    During warmup: lambda_reg=100.0 (SIGReg dominates)")

    best_val_loss = float("inf")
    use_amp = device.type == "cuda"

    # Training history for visualization
    charts_path = checkpoint_dir / "training_charts.html"
    history_path = checkpoint_dir / "training_history.json"

    # Load existing history if resuming
    training_history: List[Dict] = []
    if start_epoch > 0 and history_path.exists():
        import json
        with open(history_path, "r") as f:
            training_history = json.load(f)
        print(f"Loaded {len(training_history)} epochs from training history")
        # Restore best validation loss from history
        if training_history:
            best_val_loss = min(h.get("val_loss", float("inf")) for h in training_history)
            print(f"Best val_loss from previous training: {best_val_loss:.6f}")

    for epoch in range(start_epoch, args.epochs):
        is_repr_warmup = epoch < repr_warmup_epochs
        if is_repr_warmup:
            model.lambda_reg = 100.0
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{args.epochs} [REPR WARMUP]")
            print("=" * 70)
        else:
            model.lambda_reg = original_lambda_reg
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{args.epochs}")
            print("=" * 70)

        train_metrics, global_step = train_jepa_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            global_step=global_step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            base_lr=args.lr,
            log_every=args.log_every,
            use_amp=use_amp,
            accumulation_steps=getattr(args, 'gradient_accumulation_steps', 1),
        )

        print(f"\n  Train - Loss: {train_metrics['loss']:.4f} | "
              f"Pred: {train_metrics['pred_loss']:.4f} | "
              f"SIGReg: {train_metrics['sigreg_loss']:.4f}")

        val_metrics = validate_jepa(model, val_loader, device, args.embedding_dim, use_amp=use_amp)

        print(f"  Val   - Loss: {val_metrics['val_loss']:.4f} | "
              f"Pred: {val_metrics['val_pred_loss']:.4f}")
        effective_rank = val_metrics.get("effective_rank", 0)
        print(f"  Embedding Stats - STD: {val_metrics['embedding_std']:.4f} | "
              f"Norm: {val_metrics['embedding_mean_norm']:.4f} | "
              f"EffRank: {effective_rank:.0f}/{args.embedding_dim}")

        min_effective_rank = int(args.embedding_dim * args.min_effective_rank_pct)
        if effective_rank < min_effective_rank:
            print(f"  WARNING: Embedding collapse detected!")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history for visualization
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["val_loss"],
            "pred_loss": train_metrics.get("pred_loss"),
            "sigreg_loss": train_metrics.get("sigreg_loss"),
            "embedding_std": val_metrics.get("embedding_std"),
            "embedding_mean_norm": val_metrics.get("embedding_mean_norm"),
            "effective_rank": val_metrics.get("effective_rank"),
            "lr": current_lr,
            "is_warmup": is_repr_warmup,
        }
        training_history.append(epoch_record)

        # Generate training charts (every epoch for real-time monitoring)
        generate_training_charts(
            history=training_history,
            output_path=charts_path,
            title=f"LeJEPA Training: {args.checkpoint_id}",
            model_type="jepa",
        )
        save_training_history(training_history, history_path)

        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"lejepa_epoch_{epoch + 1:04d}.pt"
            model.save_checkpoint(
                checkpoint_path,
                epoch=epoch + 1,
                optimizer=optimizer,
                global_step=global_step,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )

        # Check embedding health
        current_variance = val_metrics.get("embedding_std", 0.0)
        is_rank_healthy = effective_rank >= min_effective_rank
        is_variance_healthy = (current_variance > 0.5) and (current_variance < 2.0)
        is_healthy = is_rank_healthy and is_variance_healthy

        if val_metrics["val_loss"] < best_val_loss:
            if is_healthy:
                best_val_loss = val_metrics["val_loss"]
                best_path = checkpoint_dir / "lejepa_best.pt"
                model.save_checkpoint(
                    best_path,
                    epoch=epoch + 1,
                    optimizer=optimizer,
                    global_step=global_step,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                )
                print(f"  New best model saved! Val loss: {best_val_loss:.4f}")
            else:
                if not is_rank_healthy:
                    print(f"  Val loss improved but collapsed - skipping")
                else:
                    print(f"  Val loss improved but variance unhealthy - skipping")

    # Save final model
    final_path = checkpoint_dir / "lejepa_final.pt"
    model.save_checkpoint(
        final_path,
        epoch=args.epochs,
        optimizer=optimizer,
        global_step=global_step,
    )

    mgr.set_training_complete(
        args.checkpoint_id,
        metrics={"best_val_loss": float(best_val_loss)},
        status="trained",
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final checkpoint: {final_path}")
    print(f"Best checkpoint: {checkpoint_dir / 'lejepa_best.pt'}")
    print(f"Training charts: {charts_path}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
