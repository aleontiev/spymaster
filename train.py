#!/usr/bin/env python3
"""
Unified training script for SpyMaster models.

This is the entry point for all model training. The actual training logic
is split by model type in the train/ package:
- train/jepa.py: LeJEPA self-supervised pre-training
- train/entry.py: Entry policy training (3-class, 5-class, regression, directional)
- train/common.py: Shared utilities

Usage:
    # Train LeJEPA
    uv run python train.py --model-type jepa --checkpoint-id lejepa-v1 --epochs 10

    # Train 3-class entry policy
    uv run python train.py --model-type entry-3class \\
        --lejepa-checkpoint checkpoints/lejepa/lejepa_best.pt \\
        --checkpoint-id entry-v1 --epochs 50

    # Train 5-class entry policy
    uv run python train.py --model-type entry-5class \\
        --lejepa-checkpoint checkpoints/lejepa/lejepa_best.pt \\
        --checkpoint-id entry-5class-v1 --epochs 50

    # Train directional move entry policy
    uv run python train.py --model-type entry-directional \\
        --lejepa-checkpoint checkpoints/lejepa/lejepa_best.pt \\
        --checkpoint-id entry-dir-v1 --epochs 50

    # Train regression entry policy
    uv run python train.py --model-type entry-regression \\
        --lejepa-checkpoint checkpoints/lejepa/lejepa_best.pt \\
        --checkpoint-id entry-reg-v1 --epochs 50

    # Pre-compute embeddings for masked gated policy
    uv run python train.py --model-type masked-gated-policy --precompute-only \\
        --short-jepa-checkpoint checkpoints/lejepa-15-5/lejepa_best.pt \\
        --med-jepa-checkpoint checkpoints/lejepa-45-15/lejepa_best.pt \\
        --long-jepa-checkpoint checkpoints/lejepa-90-30/lejepa_best.pt \\
        --checkpoint-id masked-policy-v1

    # Train masked gated policy (after embeddings are pre-computed)
    uv run python train.py --model-type masked-gated-policy \\
        --embeddings-dir data/embeddings \\
        --checkpoint-id masked-policy-v1 --epochs 50

    # Train multi-scale policy with MAE labels (requires precomputed embeddings with MAE labels)
    uv run python train.py --model-type masked-gated-policy --multi-scale \\
        --embeddings-dir data/embeddings \\
        --checkpoint-id multiscale-policy-v1 --epochs 50 \\
        --head-weight 1.0 --combined-weight 0.5
"""

import argparse
import sys


def create_base_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """Create the base argument parser with common arguments."""
    parser = argparse.ArgumentParser(
        description="Unified training script for LeJEPA and entry policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=add_help,
    )

    # Model type (required)
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["jepa", "entry-3class", "entry-5class", "entry-regression", "entry-directional", "entry-mae", "entry-percentile", "masked-gated-policy"],
        help="Type of model to train",
    )

    # Checkpoint management
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        required=True,
        help="Checkpoint ID for the registry",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint name/UUID or path to resume from",
    )

    # LeJEPA checkpoint (required for entry policies)
    parser.add_argument(
        "--lejepa-checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained LeJEPA checkpoint (required for entry policies)",
    )

    # Data parameters
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory",
    )
    parser.add_argument(
        "--underlying",
        type=str,
        default="SPY",
        help="Underlying symbol (default: SPY)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for training data (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Max parquet files to load (for testing)",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=90,
        help="Context window length in minutes",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=30,
        help="Target window length in minutes (JEPA)",
    )
    parser.add_argument(
        "--prediction-gap",
        type=int,
        default=0,
        help="Gap between context end and target start (JEPA)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride (JEPA)",
    )
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Linear warmup epochs for learning rate",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training",
    )

    # System parameters
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', or 'cpu'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers",
    )

    # Logging and saving
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log every N steps/epochs",
    )

    return parser


def main() -> None:
    """Main entry point dispatching to appropriate training function."""
    # First, parse just the model type without help to know which additional args to add
    base_parser = create_base_parser(add_help=False)
    args, _ = base_parser.parse_known_args()

    # Now create full parser with help enabled and add model-specific args
    full_parser = create_base_parser(add_help=True)

    if args.model_type == "jepa":
        from train.jepa import add_jepa_args, train_jepa
        add_jepa_args(full_parser)
        args = full_parser.parse_args()
        train_jepa(args)

    elif args.model_type in ["entry-3class", "entry-5class"]:
        from train.entry import add_entry_args
        add_entry_args(full_parser)
        args = full_parser.parse_args()
        # Validate LeJEPA checkpoint requirement
        if not args.lejepa_checkpoint:
            full_parser.error(f"--lejepa-checkpoint is required for {args.model_type}")
        _train_entry_classification(args)

    elif args.model_type == "entry-regression":
        from train.entry import add_entry_args
        add_entry_args(full_parser)
        args = full_parser.parse_args()
        if not args.lejepa_checkpoint:
            full_parser.error(f"--lejepa-checkpoint is required for {args.model_type}")
        _train_entry_regression(args)

    elif args.model_type == "entry-directional":
        from train.entry import add_entry_args
        add_entry_args(full_parser)
        args = full_parser.parse_args()
        if not args.lejepa_checkpoint:
            full_parser.error(f"--lejepa-checkpoint is required for {args.model_type}")
        _train_entry_directional(args)

    elif args.model_type == "entry-mae":
        from train.entry import add_entry_args
        add_entry_args(full_parser)
        args = full_parser.parse_args()
        if not args.lejepa_checkpoint:
            full_parser.error(f"--lejepa-checkpoint is required for {args.model_type}")
        _train_entry_mae(args)

    elif args.model_type == "entry-percentile":
        from train.entry import add_entry_args
        add_entry_args(full_parser)
        args = full_parser.parse_args()
        if not args.lejepa_checkpoint:
            full_parser.error(f"--lejepa-checkpoint is required for {args.model_type}")
        _train_entry_percentile(args)

    elif args.model_type == "masked-gated-policy":
        from train.masked_policy import add_masked_policy_args, train_masked_policy, train_multiscale_policy
        add_masked_policy_args(full_parser)
        args = full_parser.parse_args()
        if getattr(args, 'multi_scale', False):
            train_multiscale_policy(args)
        else:
            train_masked_policy(args)


def _train_entry_classification(args: argparse.Namespace) -> None:
    """Train 3-class or 5-class entry policy."""
    from datetime import datetime, time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from src.data.processing import MarketPatch
    from src.data.dag.loader import load_normalized_data, RAW_CACHE_DIR, parse_date
    from src.model.lejepa import LeJEPA
    from src.model.policy import EntryPolicy
    from src.model.loss import FocalLoss
    from src.managers.checkpoint_manager import CheckpointManager

    from train.common import set_seed, get_device, EmbeddingDataset, RealOptionsROICalculator
    from train.entry import (
        Entry3ClassDataset,
        Entry5ClassDataset,
        train_entry_epoch,
        validate_entry,
        save_entry_checkpoint,
    )

    num_classes = 5 if args.model_type == "entry-5class" else 3
    class_name = "5-Class" if num_classes == 5 else "3-Class"

    print("=" * 70)
    print(f"{class_name} Entry Policy Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"\nDevice: {device}")

    # Load pre-trained LeJEPA
    print("\n" + "=" * 70)
    print("Loading Pre-trained LeJEPA")
    print("=" * 70)

    lejepa = LeJEPA.load(args.lejepa_checkpoint, map_location=str(device))
    lejepa = lejepa.to(device)
    lejepa.eval()
    for param in lejepa.parameters():
        param.requires_grad = False

    embedding_dim = lejepa.embedding_dim
    lejepa_context_len = lejepa.max_context_len
    print(f"Embedding dimension: {embedding_dim}")
    print(f"LeJEPA context length: {lejepa_context_len}")

    # Override context_len to match LeJEPA if not explicitly set
    if args.context_len != lejepa_context_len:
        print(f"  Overriding --context-len {args.context_len} -> {lejepa_context_len} (from LeJEPA)")
        args.context_len = lejepa_context_len

    # Load training data
    # - Normalized data: for LeJEPA patches
    # - Raw data: for computing actual price change labels
    print("\n" + "=" * 70)
    print("Loading Training Data")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    normalized_dir = data_dir / "training-1m-normalized" / args.underlying
    raw_dir = data_dir / "training-1m-raw" / args.underlying

    if not normalized_dir.exists():
        raise ValueError(f"Normalized data directory not found: {normalized_dir}")

    # Parse date range
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    # Detect features from most recent normalized file (has the most features)
    sample_files = sorted(normalized_dir.glob("*/*.parquet"))
    if not sample_files:
        raise ValueError(f"No parquet files found in {normalized_dir}")

    sample_df = pd.read_parquet(sample_files[-1])  # Use latest file
    exclude_cols = {'timestamp', 'date', 'datetime', 'index', 'symbol', 'underlying'}
    feature_cols = sorted([
        col for col in sample_df.columns
        if col.lower() not in exclude_cols
        and sample_df[col].dtype in ['float64', 'float32', 'int64', 'int32']
    ])

    # Verify feature count matches LeJEPA
    lejepa_input_dim = lejepa.input_dim
    if len(feature_cols) != lejepa_input_dim:
        print(f"WARNING: Normalized data has {len(feature_cols)} features, LeJEPA expects {lejepa_input_dim}")
        print(f"  Using first {lejepa_input_dim} features to match LeJEPA")
        feature_cols = feature_cols[:lejepa_input_dim]

    print(f"Using {len(feature_cols)} features")

    # Load all data files
    all_normalized_data = []
    all_raw_closes = []
    all_dates = []
    file_count = 0

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

                # Load normalized data for patches
                norm_df = pd.read_parquet(parquet_file)

                # Try to load raw data for labels
                raw_file = raw_dir / month_str / f"{day_str}.parquet"
                if raw_file.exists():
                    raw_df = pd.read_parquet(raw_file)
                    if "close" in raw_df.columns:
                        raw_close = raw_df["close"].values
                    else:
                        continue
                else:
                    # Fallback: use normalized close (less accurate for labels)
                    raw_close = norm_df["close"].values if "close" in norm_df.columns else None
                    if raw_close is None:
                        continue

                # Validate all required features are present
                missing_cols = set(feature_cols) - set(norm_df.columns)
                if missing_cols:
                    raise ValueError(
                        f"File {parquet_file} is missing {len(missing_cols)} expected columns:\n"
                        f"  {sorted(missing_cols)[:10]}{'...' if len(missing_cols) > 10 else ''}\n"
                        f"This file has {len(norm_df.columns)} columns, but canonical set has {len(feature_cols)}.\n"
                        f"You may need to regenerate training data for older dates."
                    )

                # Filter to market hours (first 390 rows)
                market_len = min(390, len(norm_df), len(raw_close))
                norm_df = norm_df.iloc[:market_len]
                raw_close = raw_close[:market_len]

                norm_df = norm_df[feature_cols].copy()
                norm_df = norm_df.fillna(0)

                all_normalized_data.append(norm_df.values.astype(np.float32))
                all_raw_closes.append(raw_close.astype(np.float32))
                all_dates.extend([file_date] * market_len)

                file_count += 1
                if args.max_files and file_count >= args.max_files:
                    break
            except Exception as e:
                continue
        if args.max_files and file_count >= args.max_files:
            break

    if not all_normalized_data:
        raise ValueError(f"No data loaded from {normalized_dir}")

    # Concatenate all data
    data_array = np.vstack(all_normalized_data)
    raw_closes = np.concatenate(all_raw_closes)
    dates_array = np.array(all_dates)

    print(f"Loaded {len(data_array):,} bars from {file_count} days")
    print(f"Date range: {dates_array[0]} to {dates_array[-1]}")
    print(f"Features: {len(feature_cols)}")
    print(f"Raw closes range: ${raw_closes.min():.2f} - ${raw_closes.max():.2f}")

    # Create DataFrame for compatibility
    df = pd.DataFrame(data_array, columns=feature_cols)
    df["date"] = dates_array

    # Split by days
    unique_dates = np.unique(dates_array)
    n_days = len(unique_dates)
    n_train_days = int(n_days * args.train_split)
    train_days = set(unique_dates[:n_train_days])
    val_days = set(unique_dates[n_train_days:])

    train_mask = np.array([d in train_days for d in dates_array])
    val_mask = np.array([d in val_days for d in dates_array])

    train_df = df[train_mask].drop(columns=["date"])
    val_df = df[val_mask].drop(columns=["date"])
    train_raw_closes = raw_closes[train_mask]
    val_raw_closes = raw_closes[val_mask]

    print(f"Total days: {n_days}, Train: {len(train_days)}, Val: {len(val_days)}")

    # Create datasets
    if num_classes == 5:
        base_cols = {"open", "high", "low", "close", "volume"}
        extra_columns = [col for col in train_df.columns if col not in base_cols]
        patcher = MarketPatch(patch_length=args.context_len, extra_columns=extra_columns)

        roi_calculator = RealOptionsROICalculator(
            options_dir=str(options_dir),
            base_iv=args.base_iv,
            execution_delay_minutes=args.execution_delay,
            slippage_pct=args.slippage_pct,
        )

        train_base_dataset = Entry5ClassDataset(
            df=train_df, patcher=patcher, roi_calculator=roi_calculator,
            context_len=args.context_len, lookahead=args.lookahead,
            min_roi_threshold=args.min_roi_threshold, otm_buffer=args.otm_buffer,
            atm_offset=args.atm_offset, otm_offset=args.otm_offset,
            no_time_decay=args.no_time_decay,
        )
        val_base_dataset = Entry5ClassDataset(
            df=val_df, patcher=patcher, roi_calculator=roi_calculator,
            context_len=args.context_len, lookahead=args.lookahead,
            min_roi_threshold=args.min_roi_threshold, otm_buffer=args.otm_buffer,
            atm_offset=args.atm_offset, otm_offset=args.otm_offset,
            no_time_decay=args.no_time_decay,
        )
    else:
        train_base_dataset = Entry3ClassDataset(
            normalized_df=train_df, raw_closes=train_raw_closes,
            context_len=args.context_len, lookahead=args.lookahead,
            threshold_pct=args.threshold_pct,
        )
        val_base_dataset = Entry3ClassDataset(
            normalized_df=val_df, raw_closes=val_raw_closes,
            context_len=args.context_len, lookahead=args.lookahead,
            threshold_pct=args.threshold_pct,
        )

    train_dist = train_base_dataset.get_class_distribution()
    val_dist = val_base_dataset.get_class_distribution()
    print(f"\nTrain samples: {len(train_base_dataset):,}")
    print(f"Val samples: {len(val_base_dataset):,}")

    # Pre-compute embeddings
    train_dataset = EmbeddingDataset(train_base_dataset, lejepa, device)
    val_dataset = EmbeddingDataset(val_base_dataset, lejepa, device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Initialize policy
    policy = EntryPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=args.policy_hidden_dim,
        num_layers=args.policy_layers,
        dropout=args.dropout,
        num_actions=num_classes,
    )
    policy = policy.to(device)

    # Loss function
    if args.use_focal_loss:
        if num_classes == 5:
            focal_alpha = torch.tensor([
                args.focal_alpha_hold, args.focal_alpha_atm, args.focal_alpha_otm,
                args.focal_alpha_atm, args.focal_alpha_otm,
            ])
        else:
            focal_alpha = torch.tensor([
                args.focal_alpha_hold, args.focal_alpha_signal, args.focal_alpha_signal,
            ])
        criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr / 100)

    # Checkpoint management
    project_root = Path(__file__).parent
    mgr = CheckpointManager(project_root / "checkpoints", project_root)
    entry = mgr.get_or_create(args.checkpoint_id, model_type=args.model_type, config={})
    checkpoint_dir = Path(entry["path"])
    print(f"\nCheckpoint directory: {checkpoint_dir}")

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * epoch / args.warmup_epochs

        train_metrics = train_entry_epoch(policy, train_loader, optimizer, criterion, device, num_classes)
        val_metrics = validate_entry(policy, val_loader, criterion, device, num_classes)

        if epoch > args.warmup_epochs:
            scheduler.step()

        if epoch % args.log_every == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train: Loss={train_metrics['train_loss']:.4f} | Acc={train_metrics['train_accuracy']:.1%}")
            print(f"  Val:   Loss={val_metrics['val_loss']:.4f} | Acc={val_metrics['val_accuracy']:.1%}")

        if val_metrics["val_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            save_entry_checkpoint(policy, optimizer, epoch, {**train_metrics, **val_metrics},
                                  checkpoint_dir / "entry_policy_best.pt", embedding_dim, args)
            print(f"  New best! Val Acc: {best_val_acc:.1%}")

    save_entry_checkpoint(policy, optimizer, args.epochs, {**train_metrics, **val_metrics},
                          checkpoint_dir / "entry_policy_final.pt", embedding_dim, args)

    mgr.set_training_complete(args.checkpoint_id, metrics={"best_val_accuracy": float(best_val_acc)}, status="trained")
    print(f"\nTraining Complete! Best val accuracy: {best_val_acc:.1%}")


def _train_entry_regression(args: argparse.Namespace) -> None:
    """Train regression entry policy."""
    from datetime import datetime, time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from src.data.dag.loader import load_normalized_data, RAW_CACHE_DIR, parse_date
    from src.model.lejepa import LeJEPA
    from src.model.policy import RegressionEntryPolicy
    from src.managers.checkpoint_manager import CheckpointManager

    from train.common import set_seed, get_device, RegressionEmbeddingDataset
    from train.entry import (
        EntryRegressionDataset,
        train_regression_epoch,
        validate_regression,
        save_regression_checkpoint,
    )

    print("=" * 70)
    print("Regression Entry Policy Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seed(args.seed)
    device = get_device(args.device)

    # Load LeJEPA
    lejepa = LeJEPA.load(args.lejepa_checkpoint, map_location=str(device))
    lejepa = lejepa.to(device)
    lejepa.eval()
    for param in lejepa.parameters():
        param.requires_grad = False

    embedding_dim = lejepa.embedding_dim

    # Load data (similar to classification)
    data_dir = Path(args.data_dir)
    stocks_dir = data_dir / "stocks-1m"
    options_dir = data_dir / "options-1m"

    df = load_normalized_data(
        stocks_dir=str(stocks_dir), options_dir=str(options_dir),
        underlying=args.underlying, start_date=args.start_date,
        end_date=args.end_date, max_files=args.max_files,
    )

    # Load raw closes
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    raw_cache_dir = Path(RAW_CACHE_DIR)
    raw_files = sorted(raw_cache_dir.glob(f"{args.underlying}_raw_*.parquet"))
    raw_closes_list = []

    for rf in raw_files:
        date_str = rf.stem.split("_")[-1]
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if start and file_date < start:
                continue
            if end and file_date > end:
                continue
            raw_df = pd.read_parquet(rf)
            raw_closes_list.append(raw_df["close"].values)
        except (ValueError, KeyError):
            continue

    raw_closes = np.concatenate(raw_closes_list) if raw_closes_list else np.zeros(len(df))

    if len(raw_closes) != len(df):
        min_len = min(len(raw_closes), len(df))
        raw_closes = raw_closes[:min_len]
        df = df.iloc[:min_len]

    # Filter to market hours
    market_open_utc = time(14, 30)
    market_close_utc = time(21, 0)
    mask = (df.index.time >= market_open_utc) & (df.index.time < market_close_utc)
    mask_array = np.array(mask)
    df = df[mask]
    raw_closes = raw_closes[mask_array]

    # Split
    df["date"] = df.index.date
    days = df["date"].unique()
    n_train_days = int(len(days) * args.train_split)
    train_days = days[:n_train_days]
    val_days = days[n_train_days:]

    train_mask = df["date"].isin(train_days)
    val_mask = df["date"].isin(val_days)

    train_df = df[train_mask].drop(columns=["date"])
    val_df = df[val_mask].drop(columns=["date"])
    train_raw_closes = raw_closes[train_mask]
    val_raw_closes = raw_closes[val_mask]

    # Create datasets
    train_base_dataset = EntryRegressionDataset(
        train_df, train_raw_closes, args.context_len, args.lookahead, args.scale_factor
    )
    val_base_dataset = EntryRegressionDataset(
        val_df, val_raw_closes, args.context_len, args.lookahead, args.scale_factor
    )

    train_dataset = RegressionEmbeddingDataset(train_base_dataset, lejepa, device)
    val_dataset = RegressionEmbeddingDataset(val_base_dataset, lejepa, device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Initialize policy
    policy = RegressionEntryPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=args.policy_hidden_dim,
        num_layers=args.policy_layers,
        dropout=args.dropout,
    )
    policy = policy.to(device)

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Checkpoint
    project_root = Path(__file__).parent
    mgr = CheckpointManager(project_root / "checkpoints", project_root)
    entry = mgr.get_or_create(args.checkpoint_id, model_type="entry-regression", config={})
    checkpoint_dir = Path(entry["path"])

    # Training loop
    best_val_corr = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_regression_epoch(policy, train_loader, optimizer, device)
        val_metrics = validate_regression(policy, val_loader, device)
        scheduler.step()

        if epoch % args.log_every == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train: Loss={train_metrics['train_loss']:.4f} | Corr={train_metrics['train_correlation']:.4f}")
            print(f"  Val:   Loss={val_metrics['val_loss']:.4f} | Corr={val_metrics['val_correlation']:.4f}")

        if val_metrics["val_correlation"] > best_val_corr:
            best_val_corr = val_metrics["val_correlation"]
            save_regression_checkpoint(policy, optimizer, epoch, {**train_metrics, **val_metrics},
                                       checkpoint_dir / "regression_policy_best.pt", embedding_dim, args)
            print(f"  New best! Val Corr: {best_val_corr:.4f}")

    save_regression_checkpoint(policy, optimizer, args.epochs, {**train_metrics, **val_metrics},
                               checkpoint_dir / "regression_policy_final.pt", embedding_dim, args)
    mgr.set_training_complete(args.checkpoint_id, metrics={"best_val_correlation": float(best_val_corr)}, status="trained")
    print(f"\nTraining Complete! Best val correlation: {best_val_corr:.4f}")


def _train_entry_directional(args: argparse.Namespace) -> None:
    """Train directional move entry policy."""
    from datetime import datetime, time
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from src.data.processing import MarketPatch
    from src.data.dag.loader import load_normalized_data, parse_date
    from src.model.lejepa import LeJEPA
    from src.model.policy import EntryPolicy
    from src.model.loss import FocalLoss
    from src.managers.checkpoint_manager import CheckpointManager

    from train.common import set_seed, get_device, EmbeddingDataset, RealOptionsROICalculator
    from train.entry import (
        DirectionalMoveDataset,
        train_entry_epoch,
        validate_entry,
        save_entry_checkpoint,
    )

    num_classes = 5

    print("=" * 70)
    print("Directional Move Entry Policy Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seed(args.seed)
    device = get_device(args.device)

    # Load LeJEPA
    lejepa = LeJEPA.load(args.lejepa_checkpoint, map_location=str(device))
    lejepa = lejepa.to(device)
    lejepa.eval()
    for param in lejepa.parameters():
        param.requires_grad = False

    embedding_dim = lejepa.embedding_dim

    # Load data
    data_dir = Path(args.data_dir)
    stocks_dir = data_dir / "stocks-1m"
    options_dir = data_dir / "options-1m"
    raw_dir = data_dir / "training-1m-raw" / args.underlying

    df = load_normalized_data(
        stocks_dir=str(stocks_dir), options_dir=str(options_dir),
        underlying=args.underlying, start_date=args.start_date,
        end_date=args.end_date, max_files=args.max_files,
    )

    # Load raw OHLCV
    raw_files = sorted(raw_dir.glob("**/*.parquet"))
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)

    raw_dfs = []
    for rf in raw_files:
        try:
            month_dir = rf.parent.name
            day = rf.stem
            file_date = datetime.strptime(f"{month_dir}-{day}", "%Y-%m-%d").date()
            if start and file_date < start:
                continue
            if end and file_date > end:
                continue
            raw_dfs.append(pd.read_parquet(rf))
        except (ValueError, KeyError):
            continue

    raw_df = pd.concat(raw_dfs, ignore_index=False).sort_index() if raw_dfs else None
    if raw_df is None:
        raise ValueError(f"No raw OHLCV data found in {raw_dir}")

    # Filter to market hours
    market_open_utc = time(14, 30)
    market_close_utc = time(21, 0)
    mask = (df.index.time >= market_open_utc) & (df.index.time < market_close_utc)
    df = df[mask]
    raw_mask = (raw_df.index.time >= market_open_utc) & (raw_df.index.time < market_close_utc)
    raw_df = raw_df[raw_mask]

    # Align
    if len(df) != len(raw_df):
        common_idx = df.index.intersection(raw_df.index)
        df = df.loc[common_idx]
        raw_df = raw_df.loc[common_idx]

    # Split
    df["date"] = df.index.date
    raw_df["date"] = raw_df.index.date
    days = df["date"].unique()
    n_train_days = int(len(days) * args.train_split)
    train_days = days[:n_train_days]
    val_days = days[n_train_days:]

    train_mask = df["date"].isin(train_days)
    val_mask = df["date"].isin(val_days)

    train_df = df[train_mask].drop(columns=["date"])
    val_df = df[val_mask].drop(columns=["date"])

    train_raw_df = raw_df[raw_df["date"].isin(train_days)]
    val_raw_df = raw_df[raw_df["date"].isin(val_days)]

    train_raw_ohlcv = {k: train_raw_df[k].values for k in ["open", "high", "low", "close"]}
    val_raw_ohlcv = {k: val_raw_df[k].values for k in ["open", "high", "low", "close"]}

    # Create datasets
    base_cols = {"open", "high", "low", "close", "volume"}
    extra_columns = [col for col in train_df.columns if col not in base_cols]
    patcher = MarketPatch(patch_length=args.context_len, extra_columns=extra_columns)

    roi_calculator = RealOptionsROICalculator(
        options_dir=str(options_dir),
        base_iv=args.base_iv,
        execution_delay_minutes=args.execution_delay,
        slippage_pct=args.slippage_pct,
    )

    train_base_dataset = DirectionalMoveDataset(
        normalized_df=train_df, raw_ohlcv=train_raw_ohlcv, patcher=patcher,
        roi_calculator=roi_calculator, context_len=args.context_len,
        candle_agg_minutes=args.candle_agg_minutes, min_candles=args.min_candles,
        max_candles=args.max_candles, min_move_pct=args.min_move_pct,
        min_consistency_pct=args.min_consistency_pct,
        wick_penalty_weight=args.wick_penalty_weight, move_size_weight=args.move_size_weight,
        atm_offset=args.atm_offset, otm_offset=args.otm_offset,
        otm_buffer=args.otm_buffer, min_roi_threshold=args.min_roi_threshold,
    )
    val_base_dataset = DirectionalMoveDataset(
        normalized_df=val_df, raw_ohlcv=val_raw_ohlcv, patcher=patcher,
        roi_calculator=roi_calculator, context_len=args.context_len,
        candle_agg_minutes=args.candle_agg_minutes, min_candles=args.min_candles,
        max_candles=args.max_candles, min_move_pct=args.min_move_pct,
        min_consistency_pct=args.min_consistency_pct,
        wick_penalty_weight=args.wick_penalty_weight, move_size_weight=args.move_size_weight,
        atm_offset=args.atm_offset, otm_offset=args.otm_offset,
        otm_buffer=args.otm_buffer, min_roi_threshold=args.min_roi_threshold,
    )

    train_dist = train_base_dataset.get_class_distribution()
    val_dist = val_base_dataset.get_class_distribution()
    print(f"\nTrain samples: {len(train_base_dataset):,}")
    print(f"Val samples: {len(val_base_dataset):,}")

    train_dataset = EmbeddingDataset(train_base_dataset, lejepa, device)
    val_dataset = EmbeddingDataset(val_base_dataset, lejepa, device)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Initialize policy
    policy = EntryPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=args.policy_hidden_dim,
        num_layers=args.policy_layers,
        dropout=args.dropout,
        num_actions=num_classes,
    )
    policy = policy.to(device)

    # Loss function
    if args.use_focal_loss:
        focal_alpha = torch.tensor([
            args.focal_alpha_hold, args.focal_alpha_atm, args.focal_alpha_otm,
            args.focal_alpha_atm, args.focal_alpha_otm,
        ])
        criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr / 100)

    # Checkpoint
    project_root = Path(__file__).parent
    mgr = CheckpointManager(project_root / "checkpoints", project_root)
    entry = mgr.get_or_create(args.checkpoint_id, model_type=args.model_type, config={})
    checkpoint_dir = Path(entry["path"])
    mgr.set_label_distribution(args.checkpoint_id, train_dist, val_dist)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * epoch / args.warmup_epochs

        train_metrics = train_entry_epoch(policy, train_loader, optimizer, criterion, device, num_classes)
        val_metrics = validate_entry(policy, val_loader, criterion, device, num_classes)

        if epoch > args.warmup_epochs:
            scheduler.step()

        if epoch % args.log_every == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train: Loss={train_metrics['train_loss']:.4f} | Acc={train_metrics['train_accuracy']:.1%}")
            print(f"  Val:   Loss={val_metrics['val_loss']:.4f} | Acc={val_metrics['val_accuracy']:.1%}")

        if val_metrics["val_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            save_entry_checkpoint(policy, optimizer, epoch, {**train_metrics, **val_metrics},
                                  checkpoint_dir / "entry_policy_best.pt", embedding_dim, args)
            print(f"  New best! Val Acc: {best_val_acc:.1%}")

    save_entry_checkpoint(policy, optimizer, args.epochs, {**train_metrics, **val_metrics},
                          checkpoint_dir / "entry_policy_final.pt", embedding_dim, args)
    mgr.set_training_complete(args.checkpoint_id, metrics={"best_val_accuracy": float(best_val_acc)}, status="trained")
    print(f"\nTraining Complete! Best val accuracy: {best_val_acc:.1%}")


def _train_entry_mae(args: argparse.Namespace) -> None:
    """
    Train 3-class entry policy using MAE (Maximum Adverse Excursion) labeling.

    MAE labeling ensures only sharp, one-directional moves are labeled as LONG/SHORT:
    - LONG: Price rises >= min_move_pct with pullback < max_mae_pct
    - SHORT: Price falls >= min_move_pct with rally < max_mae_pct
    - HOLD: Everything else
    """
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from src.model.lejepa import LeJEPA
    from src.model.policy import EntryPolicy
    from src.model.loss import FocalLoss
    from src.managers.checkpoint_manager import CheckpointManager

    from train.common import set_seed, get_device, EmbeddingDataset
    from train.entry import (
        EntryMAE3ClassDataset,
        train_entry_epoch,
        validate_entry,
        save_entry_checkpoint,
    )

    num_classes = 3

    print("=" * 70)
    print("MAE-Based 3-Class Entry Policy Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nMAE Parameters:")
    print(f"  min_move_pct: {args.min_move_pct}%")
    print(f"  max_mae_pct: {args.max_mae_pct}%")
    print(f"  lookahead: {args.lookahead} bars")
    print(f"  use_close_to_close: {args.use_close_to_close}")

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"\nDevice: {device}")

    # Load pre-trained LeJEPA
    print("\n" + "=" * 70)
    print("Loading Pre-trained LeJEPA")
    print("=" * 70)

    lejepa = LeJEPA.load(args.lejepa_checkpoint, map_location=str(device))
    lejepa = lejepa.to(device)
    lejepa.eval()
    for param in lejepa.parameters():
        param.requires_grad = False

    embedding_dim = lejepa.embedding_dim
    lejepa_context_len = lejepa.max_context_len
    print(f"Embedding dimension: {embedding_dim}")
    print(f"LeJEPA context length: {lejepa_context_len}")

    # Override context_len to match LeJEPA
    if args.context_len != lejepa_context_len:
        print(f"  Overriding --context-len {args.context_len} -> {lejepa_context_len} (from LeJEPA)")
        args.context_len = lejepa_context_len

    # Load training data
    print("\n" + "=" * 70)
    print("Loading Training Data")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    normalized_dir = data_dir / "training-1m-normalized" / args.underlying
    raw_dir = data_dir / "training-1m-raw" / args.underlying

    if not normalized_dir.exists():
        raise ValueError(f"Normalized data directory not found: {normalized_dir}")

    # Parse date range
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    # Detect features from a file within the date range (not the most recent which may differ)
    sample_files = sorted(normalized_dir.glob("*/*.parquet"))
    if not sample_files:
        raise ValueError(f"No parquet files found in {normalized_dir}")

    # Find a sample file within the date range
    sample_file = None
    for f in sample_files:
        try:
            month_str = f.parent.name
            day_str = f.stem
            file_date = datetime.strptime(f"{month_str}-{day_str}", "%Y-%m-%d").date()
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            sample_file = f
            break
        except ValueError:
            continue

    if sample_file is None:
        raise ValueError(f"No parquet files found in date range {start_date} to {end_date}")

    sample_df = pd.read_parquet(sample_file)
    print(f"Using sample file: {sample_file} ({len(sample_df.columns)} columns)")
    exclude_cols = {'timestamp', 'date', 'datetime', 'index', 'symbol', 'underlying'}
    feature_cols = sorted([
        col for col in sample_df.columns
        if col.lower() not in exclude_cols
        and sample_df[col].dtype in ['float64', 'float32', 'int64', 'int32']
    ])

    # Verify feature count matches LeJEPA
    lejepa_input_dim = lejepa.input_dim
    if len(feature_cols) != lejepa_input_dim:
        print(f"WARNING: Normalized data has {len(feature_cols)} features, LeJEPA expects {lejepa_input_dim}")
        print(f"  Using first {lejepa_input_dim} features to match LeJEPA")
        feature_cols = feature_cols[:lejepa_input_dim]

    print(f"Using {len(feature_cols)} features")

    # Load all data files
    all_normalized_data = []
    all_raw_ohlc = {"open": [], "high": [], "low": [], "close": []}
    all_dates = []
    file_count = 0

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

                # Load normalized data
                norm_df = pd.read_parquet(parquet_file)

                # Load raw data for OHLC
                raw_file = raw_dir / month_str / f"{day_str}.parquet"
                if not raw_file.exists():
                    continue

                raw_df = pd.read_parquet(raw_file)
                required_cols = ["open", "high", "low", "close"]
                if not all(col in raw_df.columns for col in required_cols):
                    continue

                # Validate features
                missing_cols = set(feature_cols) - set(norm_df.columns)
                if missing_cols:
                    continue

                # Filter to market hours (first 390 rows)
                market_len = min(390, len(norm_df), len(raw_df))
                norm_df = norm_df.iloc[:market_len]
                raw_df = raw_df.iloc[:market_len]

                norm_df = norm_df[feature_cols].copy()
                norm_df = norm_df.fillna(0)

                all_normalized_data.append(norm_df.values.astype(np.float32))
                for col in required_cols:
                    all_raw_ohlc[col].append(raw_df[col].values.astype(np.float32))
                all_dates.extend([file_date] * market_len)

                file_count += 1
                if args.max_files and file_count >= args.max_files:
                    break
            except Exception as e:
                continue
        if args.max_files and file_count >= args.max_files:
            break

    if not all_normalized_data:
        raise ValueError(f"No data loaded from {normalized_dir}")

    # Concatenate all data
    data_array = np.vstack(all_normalized_data)
    raw_ohlc = {k: np.concatenate(v) for k, v in all_raw_ohlc.items()}
    dates_array = np.array(all_dates)

    print(f"Loaded {len(data_array):,} bars from {file_count} days")
    print(f"Date range: {dates_array[0]} to {dates_array[-1]}")
    print(f"Features: {len(feature_cols)}")
    print(f"Raw close range: ${raw_ohlc['close'].min():.2f} - ${raw_ohlc['close'].max():.2f}")

    # Create DataFrame
    df = pd.DataFrame(data_array, columns=feature_cols)
    df["date"] = dates_array

    # Split by days
    unique_dates = np.unique(dates_array)
    n_days = len(unique_dates)
    n_train_days = int(n_days * args.train_split)
    train_days = set(unique_dates[:n_train_days])
    val_days = set(unique_dates[n_train_days:])

    train_mask = np.array([d in train_days for d in dates_array])
    val_mask = np.array([d in val_days for d in dates_array])

    train_df = df[train_mask].drop(columns=["date"])
    val_df = df[val_mask].drop(columns=["date"])
    train_raw_ohlc = {k: v[train_mask] for k, v in raw_ohlc.items()}
    val_raw_ohlc = {k: v[val_mask] for k, v in raw_ohlc.items()}

    print(f"Total days: {n_days}, Train: {len(train_days)}, Val: {len(val_days)}")

    # Create MAE datasets
    print("\n" + "=" * 70)
    print("Creating MAE-Labeled Datasets")
    print("=" * 70)

    train_base_dataset = EntryMAE3ClassDataset(
        normalized_df=train_df,
        raw_ohlcv=train_raw_ohlc,
        context_len=args.context_len,
        lookahead=args.lookahead,
        min_move_pct=args.min_move_pct,
        max_mae_pct=args.max_mae_pct,
        use_close_to_close=args.use_close_to_close,
    )
    val_base_dataset = EntryMAE3ClassDataset(
        normalized_df=val_df,
        raw_ohlcv=val_raw_ohlc,
        context_len=args.context_len,
        lookahead=args.lookahead,
        min_move_pct=args.min_move_pct,
        max_mae_pct=args.max_mae_pct,
        use_close_to_close=args.use_close_to_close,
    )

    train_dist = train_base_dataset.get_class_distribution()
    val_dist = val_base_dataset.get_class_distribution()
    train_stats = train_base_dataset.get_label_stats()

    print(f"\nTrain samples: {len(train_base_dataset):,}")
    print(f"  HOLD: {train_dist.get('HOLD', 0):,} ({train_stats['hold_pct']:.1f}%)")
    print(f"  LONG (BUY_CALL): {train_dist.get('BUY_CALL', 0):,} ({train_stats['long_pct']:.1f}%)")
    print(f"  SHORT (BUY_PUT): {train_dist.get('BUY_PUT', 0):,} ({train_stats['short_pct']:.1f}%)")
    print(f"  Signal rate: {train_stats['signal_pct']:.1f}%")

    print(f"\nVal samples: {len(val_base_dataset):,}")

    # Pre-compute embeddings
    print("\n" + "=" * 70)
    print("Pre-computing LeJEPA Embeddings")
    print("=" * 70)

    train_dataset = EmbeddingDataset(train_base_dataset, lejepa, device)
    val_dataset = EmbeddingDataset(val_base_dataset, lejepa, device)

    # Use class-balanced sampling if requested
    if getattr(args, 'balanced_sampling', False):
        from torch.utils.data import WeightedRandomSampler
        train_labels = train_base_dataset.labels
        class_counts = np.bincount(train_labels, minlength=num_classes)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=True
        )
        print(f"\nUsing class-balanced sampling:")
        print(f"  Class counts: HOLD={class_counts[0]}, LONG={class_counts[1]}, SHORT={class_counts[2]}")
        print(f"  Class weights: HOLD={class_weights[0]:.4f}, LONG={class_weights[1]:.4f}, SHORT={class_weights[2]:.4f}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Initialize policy
    print("\n" + "=" * 70)
    print("Initializing Policy Network")
    print("=" * 70)

    policy = EntryPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=args.policy_hidden_dim,
        num_layers=args.policy_layers,
        dropout=args.dropout,
        num_actions=num_classes,
    )
    policy = policy.to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function with class weighting for imbalanced data
    if args.use_focal_loss:
        focal_alpha = torch.tensor([
            args.focal_alpha_hold,
            args.focal_alpha_signal,
            args.focal_alpha_signal,
        ])
        criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
        print(f"\nUsing Focal Loss (gamma={args.focal_gamma}, alpha={focal_alpha.tolist()})")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print("\nUsing CrossEntropyLoss")

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr / 100)

    # Checkpoint management
    project_root = Path(__file__).parent
    mgr = CheckpointManager(project_root / "data" / "checkpoints", project_root)
    entry = mgr.get_or_create(args.checkpoint_id, model_type="entry-mae", config={
        "lejepa_checkpoint": args.lejepa_checkpoint,
        "lookahead": args.lookahead,
        "min_move_pct": args.min_move_pct,
        "max_mae_pct": args.max_mae_pct,
        "use_close_to_close": args.use_close_to_close,
    })
    checkpoint_dir = Path(entry["path"])
    mgr.set_label_distribution(args.checkpoint_id, train_dist, val_dist)
    print(f"\nCheckpoint directory: {checkpoint_dir}")

    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    best_val_acc = 0.0
    best_val_signal_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        # Warmup
        if epoch <= args.warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * epoch / args.warmup_epochs

        train_metrics = train_entry_epoch(policy, train_loader, optimizer, criterion, device, num_classes)
        val_metrics = validate_entry(policy, val_loader, criterion, device, num_classes)

        if epoch > args.warmup_epochs:
            scheduler.step()

        # Compute F1 for signal classes
        # Note: train_entry_epoch uses EntryAction enum, so for 3-class the keys are:
        # buy_call_atm (value=1) and buy_call_otm (value=2) from EntryAction, not EntryActionLegacy
        precision_call = val_metrics.get("val_precision_buy_call_atm", 0)
        recall_call = val_metrics.get("val_recall_buy_call_atm", 0)
        precision_put = val_metrics.get("val_precision_buy_call_otm", 0)  # Actually mapped to BUY_PUT in legacy
        recall_put = val_metrics.get("val_recall_buy_call_otm", 0)

        f1_call = 2 * precision_call * recall_call / (precision_call + recall_call + 1e-8)
        f1_put = 2 * precision_put * recall_put / (precision_put + recall_put + 1e-8)
        signal_f1 = (f1_call + f1_put) / 2

        if epoch % args.log_every == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train: Loss={train_metrics['train_loss']:.4f} | Acc={train_metrics['train_accuracy']:.1%}")
            print(f"  Val:   Loss={val_metrics['val_loss']:.4f} | Acc={val_metrics['val_accuracy']:.1%}")
            print(f"  Val Signal F1: {signal_f1:.3f} (LONG: {f1_call:.3f}, SHORT: {f1_put:.3f})")
            if "val_recall_buy_call" in val_metrics:
                print(f"  LONG - P: {precision_call:.2%} R: {recall_call:.2%}")
            if "val_recall_buy_put" in val_metrics:
                print(f"  SHORT - P: {precision_put:.2%} R: {recall_put:.2%}")

        # Save best by signal F1 (more relevant than accuracy for imbalanced data)
        if signal_f1 > best_val_signal_f1:
            best_val_signal_f1 = signal_f1
            best_val_acc = val_metrics["val_accuracy"]
            save_entry_checkpoint(
                policy, optimizer, epoch,
                {**train_metrics, **val_metrics, "signal_f1": signal_f1},
                checkpoint_dir / "entry_policy_best.pt",
                embedding_dim, args
            )
            print(f"  New best! Signal F1: {best_val_signal_f1:.3f}")

    save_entry_checkpoint(
        policy, optimizer, args.epochs,
        {**train_metrics, **val_metrics, "signal_f1": signal_f1},
        checkpoint_dir / "entry_policy_final.pt",
        embedding_dim, args
    )

    mgr.set_training_complete(
        args.checkpoint_id,
        metrics={"best_val_accuracy": float(best_val_acc), "best_signal_f1": float(best_val_signal_f1)},
        status="trained"
    )
    print(f"\nTraining Complete!")
    print(f"  Best Val Accuracy: {best_val_acc:.1%}")
    print(f"  Best Signal F1: {best_val_signal_f1:.3f}")


def _train_entry_percentile(args: argparse.Namespace) -> None:
    """
    Train 3-class entry policy using percentile-based thresholds.

    Uses asymmetric thresholds derived from historical percentile analysis:
    - LONG: 15m return >= long_threshold_pct (default: 0.363% = 99th percentile)
    - SHORT: 15m return <= -short_threshold_pct (default: 0.400% = 1st percentile)
    - HOLD: everything else

    Expected ~3.7 signals/day per direction = ~7-8 total signals/day.
    """
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from src.model.lejepa import LeJEPA
    from src.model.policy import EntryPolicy
    from src.model.loss import FocalLoss
    from src.managers.checkpoint_manager import CheckpointManager

    from train.common import set_seed, get_device, EmbeddingDataset
    from train.entry import (
        EntryPercentile3ClassDataset,
        train_entry_epoch,
        validate_entry,
        save_entry_checkpoint,
    )

    num_classes = 3

    print("=" * 70)
    print("Percentile-Based 3-Class Entry Policy Training")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nPercentile Thresholds (15m lookahead):")
    print(f"  LONG threshold:  >= +{args.long_threshold_pct}%")
    print(f"  SHORT threshold: <= -{args.short_threshold_pct}%")
    print(f"  Expected signals: ~7-8 per day total")

    set_seed(args.seed)
    device = get_device(args.device)
    print(f"\nDevice: {device}")

    # Load pre-trained LeJEPA
    print("\n" + "=" * 70)
    print("Loading Pre-trained LeJEPA")
    print("=" * 70)

    lejepa = LeJEPA.load(args.lejepa_checkpoint, map_location=str(device))
    lejepa = lejepa.to(device)
    lejepa.eval()
    for param in lejepa.parameters():
        param.requires_grad = False

    embedding_dim = lejepa.embedding_dim
    lejepa_context_len = lejepa.max_context_len
    print(f"Embedding dimension: {embedding_dim}")
    print(f"LeJEPA context length: {lejepa_context_len}")

    # Override context_len to match LeJEPA
    if args.context_len != lejepa_context_len:
        print(f"  Overriding --context-len {args.context_len} -> {lejepa_context_len} (from LeJEPA)")
        args.context_len = lejepa_context_len

    # Load training data
    print("\n" + "=" * 70)
    print("Loading Training Data")
    print("=" * 70)

    data_dir = Path(args.data_dir)
    normalized_dir = data_dir / "training-1m-normalized" / args.underlying
    raw_dir = data_dir / "training-1m-raw" / args.underlying

    if not normalized_dir.exists():
        raise ValueError(f"Normalized data directory not found: {normalized_dir}")

    # Parse date range
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else None
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else None

    # Detect features from a file within the date range
    sample_files = sorted(normalized_dir.glob("*/*.parquet"))
    if not sample_files:
        raise ValueError(f"No parquet files found in {normalized_dir}")

    # Find a sample file within the date range
    sample_file = None
    for f in sample_files:
        try:
            month_str = f.parent.name
            day_str = f.stem
            file_date = datetime.strptime(f"{month_str}-{day_str}", "%Y-%m-%d").date()
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue
            sample_file = f
            break
        except ValueError:
            continue

    if sample_file is None:
        raise ValueError(f"No parquet files found in date range {start_date} to {end_date}")

    sample_df = pd.read_parquet(sample_file)
    print(f"Using sample file: {sample_file} ({len(sample_df.columns)} columns)")
    exclude_cols = {'timestamp', 'date', 'datetime', 'index', 'symbol', 'underlying'}
    feature_cols = sorted([
        col for col in sample_df.columns
        if col.lower() not in exclude_cols
        and sample_df[col].dtype in ['float64', 'float32', 'int64', 'int32']
    ])

    # Verify feature count matches LeJEPA
    lejepa_input_dim = lejepa.input_dim
    if len(feature_cols) != lejepa_input_dim:
        print(f"WARNING: Normalized data has {len(feature_cols)} features, LeJEPA expects {lejepa_input_dim}")
        print(f"  Using first {lejepa_input_dim} features to match LeJEPA")
        feature_cols = feature_cols[:lejepa_input_dim]

    print(f"Using {len(feature_cols)} features")

    # Load all data files
    all_normalized_data = []
    all_raw_closes = []
    all_dates = []
    file_count = 0

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

                # Load normalized data
                norm_df = pd.read_parquet(parquet_file)

                # Load raw data for close prices
                raw_file = raw_dir / month_str / f"{day_str}.parquet"
                if not raw_file.exists():
                    continue

                raw_df = pd.read_parquet(raw_file)
                if "close" not in raw_df.columns:
                    continue

                # Validate features
                missing_cols = set(feature_cols) - set(norm_df.columns)
                if missing_cols:
                    continue

                # Filter to market hours (first 390 rows)
                market_len = min(390, len(norm_df), len(raw_df))
                norm_df = norm_df.iloc[:market_len]
                raw_close = raw_df["close"].values[:market_len].astype(np.float32)

                norm_df = norm_df[feature_cols].copy()
                norm_df = norm_df.fillna(0)

                all_normalized_data.append(norm_df.values.astype(np.float32))
                all_raw_closes.append(raw_close)
                all_dates.extend([file_date] * market_len)

                file_count += 1
                if args.max_files and file_count >= args.max_files:
                    break
            except Exception as e:
                continue
        if args.max_files and file_count >= args.max_files:
            break

    if not all_normalized_data:
        raise ValueError(f"No data loaded from {normalized_dir}")

    # Concatenate all data
    data_array = np.vstack(all_normalized_data)
    raw_closes = np.concatenate(all_raw_closes)
    dates_array = np.array(all_dates)

    print(f"Loaded {len(data_array):,} bars from {file_count} days")
    print(f"Date range: {dates_array[0]} to {dates_array[-1]}")
    print(f"Features: {len(feature_cols)}")
    print(f"Raw closes range: ${raw_closes.min():.2f} - ${raw_closes.max():.2f}")

    # Create DataFrame
    df = pd.DataFrame(data_array, columns=feature_cols)
    df["date"] = dates_array

    # Split by days
    unique_dates = np.unique(dates_array)
    n_days = len(unique_dates)
    n_train_days = int(n_days * args.train_split)
    train_days = set(unique_dates[:n_train_days])
    val_days = set(unique_dates[n_train_days:])

    train_mask = np.array([d in train_days for d in dates_array])
    val_mask = np.array([d in val_days for d in dates_array])

    train_df = df[train_mask].drop(columns=["date"])
    val_df = df[val_mask].drop(columns=["date"])
    train_raw_closes = raw_closes[train_mask]
    val_raw_closes = raw_closes[val_mask]

    print(f"Total days: {n_days}, Train: {len(train_days)}, Val: {len(val_days)}")

    # Create percentile-based datasets
    print("\n" + "=" * 70)
    print("Creating Percentile-Labeled Datasets")
    print("=" * 70)

    train_base_dataset = EntryPercentile3ClassDataset(
        normalized_df=train_df,
        raw_closes=train_raw_closes,
        context_len=args.context_len,
        lookahead=args.lookahead,
        long_threshold_pct=args.long_threshold_pct,
        short_threshold_pct=args.short_threshold_pct,
    )
    val_base_dataset = EntryPercentile3ClassDataset(
        normalized_df=val_df,
        raw_closes=val_raw_closes,
        context_len=args.context_len,
        lookahead=args.lookahead,
        long_threshold_pct=args.long_threshold_pct,
        short_threshold_pct=args.short_threshold_pct,
    )

    train_dist = train_base_dataset.get_class_distribution()
    val_dist = val_base_dataset.get_class_distribution()
    train_stats = train_base_dataset.get_label_stats()

    print(f"\nTrain samples: {len(train_base_dataset):,}")
    print(f"  HOLD: {train_dist.get('HOLD', 0):,} ({train_stats['hold_pct']:.1f}%)")
    print(f"  LONG (BUY_CALL): {train_dist.get('BUY_CALL', 0):,} ({train_stats['long_pct']:.1f}%)")
    print(f"  SHORT (BUY_PUT): {train_dist.get('BUY_PUT', 0):,} ({train_stats['short_pct']:.1f}%)")
    print(f"  Signal rate: {train_stats['signal_pct']:.1f}%")
    print(f"  Signals per day: ~{(train_stats['long_count'] + train_stats['short_count']) / len(train_days):.1f}")

    print(f"\nVal samples: {len(val_base_dataset):,}")

    # Pre-compute embeddings
    print("\n" + "=" * 70)
    print("Pre-computing LeJEPA Embeddings")
    print("=" * 70)

    train_dataset = EmbeddingDataset(train_base_dataset, lejepa, device)
    val_dataset = EmbeddingDataset(val_base_dataset, lejepa, device)

    # Use class-balanced sampling if requested
    if getattr(args, 'balanced_sampling', False):
        from torch.utils.data import WeightedRandomSampler
        train_labels = train_base_dataset.labels
        class_counts = np.bincount(train_labels, minlength=num_classes)
        class_weights_sampling = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights_sampling[train_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_labels),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler, pin_memory=True
        )
        print(f"\nUsing class-balanced sampling:")
        print(f"  Class counts: HOLD={class_counts[0]}, LONG={class_counts[1]}, SHORT={class_counts[2]}")
        print(f"  Sample weights: HOLD={class_weights_sampling[0]:.6f}, LONG={class_weights_sampling[1]:.6f}, SHORT={class_weights_sampling[2]:.6f}")
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Initialize policy
    print("\n" + "=" * 70)
    print("Initializing Policy Network")
    print("=" * 70)

    policy = EntryPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=args.policy_hidden_dim,
        num_layers=args.policy_layers,
        dropout=args.dropout,
        num_actions=num_classes,
    )
    policy = policy.to(device)

    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Compute class weights for loss function if requested
    # This helps balance LONG/SHORT predictions by weighting rare classes more
    if getattr(args, 'use_class_weights', False):
        train_labels = train_base_dataset.labels
        class_counts = np.bincount(train_labels, minlength=num_classes)
        # Inverse frequency weighting, normalized
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_classes  # Normalize to sum to num_classes
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        print(f"\nUsing inverse-frequency class weights:")
        print(f"  Class counts: HOLD={class_counts[0]}, LONG={class_counts[1]}, SHORT={class_counts[2]}")
        print(f"  Class weights: HOLD={class_weights[0]:.4f}, LONG={class_weights[1]:.4f}, SHORT={class_weights[2]:.4f}")
    else:
        class_weights_tensor = None

    # Loss function with focal loss for imbalanced data
    if args.use_focal_loss:
        if class_weights_tensor is not None:
            # Use computed class weights instead of manual focal_alpha
            criterion = FocalLoss(alpha=class_weights_tensor, gamma=args.focal_gamma)
            print(f"\nUsing Focal Loss with class weights (gamma={args.focal_gamma})")
        else:
            focal_alpha = torch.tensor([
                args.focal_alpha_hold,
                args.focal_alpha_signal,
                args.focal_alpha_signal,
            ])
            criterion = FocalLoss(alpha=focal_alpha, gamma=args.focal_gamma)
            print(f"\nUsing Focal Loss (gamma={args.focal_gamma}, alpha={focal_alpha.tolist()})")
    else:
        if class_weights_tensor is not None:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            print("\nUsing CrossEntropyLoss with class weights")
        else:
            criterion = torch.nn.CrossEntropyLoss()
            print("\nUsing CrossEntropyLoss")

    optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr / 100)

    # Checkpoint management
    project_root = Path(__file__).parent
    mgr = CheckpointManager(project_root / "data" / "checkpoints", project_root)
    entry = mgr.get_or_create(args.checkpoint_id, model_type="entry-percentile", config={
        "lejepa_checkpoint": args.lejepa_checkpoint,
        "lookahead": args.lookahead,
        "long_threshold_pct": args.long_threshold_pct,
        "short_threshold_pct": args.short_threshold_pct,
    })
    checkpoint_dir = Path(entry["path"])
    mgr.set_label_distribution(args.checkpoint_id, train_dist, val_dist)
    print(f"\nCheckpoint directory: {checkpoint_dir}")

    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    best_val_acc = 0.0
    best_val_signal_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        # Warmup
        if epoch <= args.warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * epoch / args.warmup_epochs

        train_metrics = train_entry_epoch(policy, train_loader, optimizer, criterion, device, num_classes)
        val_metrics = validate_entry(policy, val_loader, criterion, device, num_classes)

        if epoch > args.warmup_epochs:
            scheduler.step()

        # Compute F1 for signal classes
        precision_long = val_metrics.get("val_precision_buy_call_atm", 0)
        recall_long = val_metrics.get("val_recall_buy_call_atm", 0)
        precision_short = val_metrics.get("val_precision_buy_call_otm", 0)
        recall_short = val_metrics.get("val_recall_buy_call_otm", 0)

        f1_long = 2 * precision_long * recall_long / (precision_long + recall_long + 1e-8)
        f1_short = 2 * precision_short * recall_short / (precision_short + recall_short + 1e-8)
        signal_f1 = (f1_long + f1_short) / 2

        if epoch % args.log_every == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Train: Loss={train_metrics['train_loss']:.4f} | Acc={train_metrics['train_accuracy']:.1%}")
            print(f"  Val:   Loss={val_metrics['val_loss']:.4f} | Acc={val_metrics['val_accuracy']:.1%}")
            print(f"  Val Signal F1: {signal_f1:.3f} (LONG: {f1_long:.3f}, SHORT: {f1_short:.3f})")

        # Save best by signal F1
        if signal_f1 > best_val_signal_f1:
            best_val_signal_f1 = signal_f1
            best_val_acc = val_metrics["val_accuracy"]
            save_entry_checkpoint(
                policy, optimizer, epoch,
                {**train_metrics, **val_metrics, "signal_f1": signal_f1},
                checkpoint_dir / "entry_policy_best.pt",
                embedding_dim, args
            )
            print(f"  New best! Signal F1: {best_val_signal_f1:.3f}")

    save_entry_checkpoint(
        policy, optimizer, args.epochs,
        {**train_metrics, **val_metrics, "signal_f1": signal_f1},
        checkpoint_dir / "entry_policy_final.pt",
        embedding_dim, args
    )

    mgr.set_training_complete(
        args.checkpoint_id,
        metrics={"best_val_accuracy": float(best_val_acc), "best_signal_f1": float(best_val_signal_f1)},
        status="trained"
    )
    print(f"\nTraining Complete!")
    print(f"  Best Val Accuracy: {best_val_acc:.1%}")
    print(f"  Best Signal F1: {best_val_signal_f1:.3f}")


if __name__ == "__main__":
    main()
