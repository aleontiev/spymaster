#!/usr/bin/env python3
"""
Quick linear probe evaluation at multiple prediction horizons.

Tests if LeJEPA embeddings can predict price direction at 5m, 15m, 30m lookahead.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.loader import load_normalized_data
from src.model.lejepa import LeJEPA


class LinearProbe(nn.Module):
    """Simple linear classifier for direction prediction."""

    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def extract_embeddings_with_labels(
    model: LeJEPA,
    df,
    context_len: int,
    lookahead: int,
    device: torch.device,
    batch_size: int = 512,
    input_dim: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract embeddings and direction labels."""
    model.eval()

    # Get raw closes for label computation (before normalization converts to log returns)
    # We need to reconstruct raw prices from log returns
    # Actually, let's use the close column which is normalized log returns
    # A positive sum of future log returns = price went up

    # Filter to only use the features the model was trained with
    feature_cols = [c for c in df.columns]

    # If model expects fewer features, exclude features added after model was trained
    # Features added in v10+: vwap, dist_vwap, dist_ma20/50/200, vol_regime, day_of_week
    # Feature removed in v11: gamma_concentration (need to add back as zeros for old models)
    new_features = ['vwap', 'dist_vwap', 'dist_ma20', 'dist_ma50', 'dist_ma200', 'vol_regime', 'day_of_week']
    if input_dim is not None and len(feature_cols) > input_dim:
        feature_cols = [c for c in feature_cols if c not in new_features]
        print(f"  Using {len(feature_cols)} features (excluded new features)")

    # Create data tensor
    data = torch.tensor(df[feature_cols].values, dtype=torch.float32)

    # If model expects more features than we have (e.g., gamma_concentration was removed),
    # add zero columns to match expected input_dim
    if input_dim is not None and data.shape[1] < input_dim:
        missing = input_dim - data.shape[1]
        print(f"  Adding {missing} zero columns for removed features (e.g., gamma_concentration)")
        zeros = torch.zeros(len(data), missing)
        data = torch.cat([data, zeros], dim=1)

    # For labels, we'll use the 'close' column (log returns)
    # Sum of next `lookahead` log returns indicates direction
    close_idx = feature_cols.index('close')

    embeddings = []
    labels = []

    # Valid indices
    valid_start = context_len
    valid_end = len(df) - lookahead

    if valid_end <= valid_start:
        raise ValueError(f"Not enough data: need {context_len + lookahead} bars, have {len(df)}")

    indices = list(range(valid_start, valid_end))

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(indices), batch_size), desc=f"Extracting (lookahead={lookahead}m)"):
            batch_indices = indices[batch_start:batch_start + batch_size]

            # Build context patches
            patches = []
            batch_labels = []

            for idx in batch_indices:
                # Context: [idx - context_len, idx)
                patch = data[idx - context_len:idx]
                patches.append(patch)

                # Label: sum of next `lookahead` log returns
                future_returns = data[idx:idx + lookahead, close_idx].sum().item()
                label = 1 if future_returns > 0 else 0
                batch_labels.append(label)

            patches = torch.stack(patches).to(device)

            # Get embeddings (use encode method which returns pooled embedding)
            emb = model.encode(patches)
            embeddings.append(emb.cpu())
            labels.extend(batch_labels)

    return torch.cat(embeddings, dim=0), torch.tensor(labels, dtype=torch.long)


def train_probe(
    train_emb: torch.Tensor,
    train_labels: torch.Tensor,
    val_emb: torch.Tensor,
    val_labels: torch.Tensor,
    embedding_dim: int,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.01,
) -> dict:
    """Train linear probe and return metrics."""
    probe = LinearProbe(embedding_dim, num_classes=2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_emb, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    best_val_acc = 0.0

    for epoch in range(epochs):
        probe.train()
        for emb_batch, label_batch in train_loader:
            emb_batch = emb_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()
            logits = probe(emb_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_emb.to(device))
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels.to(device)).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc

    # Final metrics
    probe.eval()
    with torch.no_grad():
        train_logits = probe(train_emb.to(device))
        train_preds = train_logits.argmax(dim=1)
        train_acc = (train_preds == train_labels.to(device)).float().mean().item()

        val_logits = probe(val_emb.to(device))
        val_preds = val_logits.argmax(dim=1)

        # Per-class accuracy
        val_labels_dev = val_labels.to(device)
        class_0_mask = val_labels_dev == 0
        class_1_mask = val_labels_dev == 1

        class_0_acc = (val_preds[class_0_mask] == 0).float().mean().item() if class_0_mask.sum() > 0 else 0
        class_1_acc = (val_preds[class_1_mask] == 1).float().mean().item() if class_1_mask.sum() > 0 else 0

    return {
        "train_acc": train_acc,
        "val_acc": best_val_acc,
        "class_0_acc": class_0_acc,
        "class_1_acc": class_1_acc,
        "positive_rate": val_labels.float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Linear probe at multiple horizons")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to LeJEPA checkpoint")
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 15, 30], help="Lookahead horizons in minutes")
    parser.add_argument("--context-len", type=int, default=90, help="Context length in minutes")
    parser.add_argument("--start-date", type=str, default="2024-01-01", help="Start date for eval data")
    parser.add_argument("--end-date", type=str, default="2024-06-30", help="End date for eval data")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--probe-epochs", type=int, default=50, help="Epochs for probe training")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Horizons: {args.horizons}")

    # Load model
    print("\nLoading LeJEPA model...")
    model, config = LeJEPA.load_checkpoint(args.checkpoint, device=str(device))
    model = model.to(device)
    model.eval()

    embedding_dim = config.get("embedding_dim", 128)
    # Get input_dim from model weights since it may not be in config
    input_dim = model.input_proj.weight.shape[1]
    print(f"Embedding dim: {embedding_dim}")
    print(f"Input dim: {input_dim}")

    for param in model.parameters():
        param.requires_grad = False

    # Load data
    print(f"\nLoading data from {args.start_date} to {args.end_date}...")
    df = load_normalized_data(
        stocks_dir="data/stocks",
        options_dir="data/options",
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"Loaded {len(df):,} bars")

    # Split into train/val
    split_idx = int(len(df) * args.train_split)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    print(f"Train: {len(train_df):,} bars, Val: {len(val_df):,} bars")

    # Run probes for each horizon
    print("\n" + "=" * 70)
    print("LINEAR PROBE RESULTS")
    print("=" * 70)

    results = {}
    for horizon in args.horizons:
        print(f"\n--- Horizon: {horizon} minutes ---")

        try:
            # Extract embeddings
            train_emb, train_labels = extract_embeddings_with_labels(
                model, train_df, args.context_len, horizon, device,
                input_dim=input_dim
            )
            val_emb, val_labels = extract_embeddings_with_labels(
                model, val_df, args.context_len, horizon, device,
                input_dim=input_dim
            )

            print(f"Train samples: {len(train_emb):,}, Val samples: {len(val_emb):,}")

            # Train probe
            metrics = train_probe(
                train_emb, train_labels,
                val_emb, val_labels,
                embedding_dim, device,
                epochs=args.probe_epochs,
            )

            results[horizon] = metrics

            # Print results
            baseline = max(metrics["positive_rate"], 1 - metrics["positive_rate"])
            edge = metrics["val_acc"] - baseline

            print(f"  Train Acc: {metrics['train_acc']:.2%}")
            print(f"  Val Acc:   {metrics['val_acc']:.2%}")
            print(f"  Baseline:  {baseline:.2%}")
            print(f"  Edge:      {edge:+.2%}")
            print(f"  Down Acc:  {metrics['class_0_acc']:.2%}")
            print(f"  Up Acc:    {metrics['class_1_acc']:.2%}")

        except Exception as e:
            print(f"  Error: {e}")
            results[horizon] = None

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Horizon':<10} {'Val Acc':<10} {'Baseline':<10} {'Edge':<10} {'Status'}")
    print("-" * 50)

    for horizon in args.horizons:
        if results.get(horizon):
            m = results[horizon]
            baseline = max(m["positive_rate"], 1 - m["positive_rate"])
            edge = m["val_acc"] - baseline

            if edge > 0.03:
                status = "✓ Good"
            elif edge > 0.01:
                status = "~ Marginal"
            else:
                status = "✗ Baseline"

            print(f"{horizon}m{'':<7} {m['val_acc']:<10.2%} {baseline:<10.2%} {edge:<+10.2%} {status}")
        else:
            print(f"{horizon}m{'':<7} {'N/A':<10} {'N/A':<10} {'N/A':<10} Error")

    print("=" * 70)


if __name__ == "__main__":
    main()
