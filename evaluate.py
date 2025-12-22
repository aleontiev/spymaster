#!/usr/bin/env python3
"""
Unified Model Evaluation Script.

Auto-detects model type from checkpoint and runs appropriate evaluation:
- LeJEPA: Linear probe evaluation for direction prediction
- Entry Policy (3-class or 5-class): Classification metrics with confusion matrix
- Exit Policy: Simulated trading metrics (requires entry policy)

Usage:
    # Evaluate any checkpoint (auto-detects type)
    uv run python evaluate.py --checkpoint checkpoints/lejepa/lejepa_best.pt

    # Evaluate with specific data range
    uv run python evaluate.py --checkpoint checkpoints/entry/entry_best.pt \
        --start-date 2024-01-01 --end-date 2024-12-31

    # Force a specific model type
    uv run python evaluate.py --checkpoint my_model.pt --model-type lejepa
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data.loader import create_combined_dataset, load_normalized_data
from src.data.processing import MarketPatch
from src.model.lejepa import LeJEPA
from src.model.policy import EntryPolicy, EntryAction, ExitPolicy, ExitAction


# =============================================================================
# Model Type Detection
# =============================================================================


@dataclass
class CheckpointInfo:
    """Information extracted from a checkpoint."""
    model_type: str  # 'lejepa', 'entry', 'exit'
    config: Dict
    epoch: int
    metrics: Optional[Dict]
    num_classes: int  # For entry: 3 or 5


def detect_model_type(checkpoint: Dict) -> CheckpointInfo:
    """
    Auto-detect model type from checkpoint structure.

    Returns:
        CheckpointInfo with detected type and config
    """
    config = checkpoint.get("config", {})
    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", None)

    # Check for LeJEPA (has model_state_dict with encoder keys)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        has_encoder = any("context_encoder" in k or "target_encoder" in k for k in state_dict.keys())
        if has_encoder:
            return CheckpointInfo(
                model_type="lejepa",
                config=config,
                epoch=epoch,
                metrics=metrics,
                num_classes=2,  # Binary direction for probe
            )

    # Check for Entry Policy (has policy_state_dict with action_head, no context_dim)
    if "policy_state_dict" in checkpoint:
        config = checkpoint.get("config", {})
        if "context_dim" in config:
            # Exit policy has context_dim
            return CheckpointInfo(
                model_type="exit",
                config=config,
                epoch=epoch,
                metrics=metrics,
                num_classes=2,
            )
        else:
            # Entry policy
            num_actions = config.get("num_actions", 5)
            return CheckpointInfo(
                model_type="entry",
                config=config,
                epoch=epoch,
                metrics=metrics,
                num_classes=num_actions,
            )

    raise ValueError("Could not detect model type from checkpoint")


# =============================================================================
# Linear Probe (for LeJEPA)
# =============================================================================


class LinearProbe(nn.Module):
    """Simple linear classifier for direction prediction."""

    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def extract_embeddings_for_probe(
    model: LeJEPA,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract embeddings and labels for linear probe evaluation."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for context_batch, target_batch in tqdm(dataloader, desc="Extracting embeddings"):
            context_batch = context_batch.to(device)
            target_batch = target_batch.to(device)

            embeddings = model.context_encoder(context_batch, return_all_tokens=False)

            # Label: is last bar of target > last bar of context?
            context_last = context_batch[:, -1, 0]
            target_last = target_batch[:, -1, 0]
            labels = (target_last > context_last).long()

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def train_and_evaluate_probe(
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    embedding_dim: int,
    device: torch.device,
    epochs: int = 100,
) -> Dict[str, float]:
    """Train linear probe and return metrics."""
    probe = LinearProbe(embedding_dim, num_classes=2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_embeddings, train_labels)
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

        probe.eval()
        with torch.no_grad():
            train_logits = probe(train_embeddings.to(device))
            train_preds = train_logits.argmax(dim=1)
            train_acc = (train_preds == train_labels.to(device)).float().mean().item()

            val_logits = probe(val_embeddings.to(device))
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_labels.to(device)).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc

    # Final per-class accuracy
    val_labels_dev = val_labels.to(device)
    per_class_acc = {}
    for cls in [0, 1]:
        mask = val_labels_dev == cls
        if mask.sum() > 0:
            per_class_acc[f"class_{cls}"] = (val_preds[mask] == cls).float().mean().item()

    return {
        "train_accuracy": train_acc,
        "val_accuracy": best_val_acc,
        "val_positive_rate": val_labels.float().mean().item(),
        **per_class_acc,
    }


def evaluate_lejepa(
    checkpoint_path: str,
    device: torch.device,
    data_dir: str = "data/stocks",
    options_dir: str = "data/options",
    start_date: str = None,
    end_date: str = None,
    batch_size: int = 512,
    probe_epochs: int = 100,
) -> Dict[str, float]:
    """Evaluate LeJEPA model with linear probe."""
    print("=" * 70)
    print("LeJEPA Evaluation: Linear Probe for Direction Prediction")
    print("=" * 70)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    feature_dim = config.get("feature_dim", 22)
    embedding_dim = config.get("embedding_dim", 256)
    num_heads = config.get("num_heads", 8)
    num_layers = config.get("num_layers", 4)
    predictor_hidden_dim = config.get("predictor_hidden_dim", 1024)

    print(f"\nModel config:")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    model = LeJEPA(
        feature_dim=feature_dim,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        predictor_hidden_dim=predictor_hidden_dim,
    ).to(device)

    # Load weights (handle torch.compile prefix)
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("._orig_mod", "")
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    print("\nLoading dataset...")
    train_dataset, val_dataset, _ = create_combined_dataset(
        stocks_dir=data_dir,
        options_dir=options_dir if options_dir else None,
        start_date=start_date,
        end_date=end_date,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    print("\nExtracting embeddings...")
    train_emb, train_labels = extract_embeddings_for_probe(model, train_loader, device)
    val_emb, val_labels = extract_embeddings_for_probe(model, val_loader, device)

    print(f"  Train shape: {train_emb.shape}")
    print(f"  Val shape: {val_emb.shape}")
    print(f"  Val label distribution: {val_labels.float().mean():.2%} positive")

    print("\nTraining linear probe...")
    metrics = train_and_evaluate_probe(
        train_emb, train_labels,
        val_emb, val_labels,
        embedding_dim, device,
        epochs=probe_epochs,
    )

    return metrics


# =============================================================================
# Entry Policy Evaluation
# =============================================================================


def evaluate_entry_policy(
    checkpoint_path: str,
    lejepa_checkpoint: str,
    device: torch.device,
    data_dir: str = "data/stocks",
    options_dir: str = "data/options",
    start_date: str = None,
    end_date: str = None,
    batch_size: int = 512,
) -> Dict[str, float]:
    """Evaluate entry policy with classification metrics."""
    print("=" * 70)
    print("Entry Policy Evaluation: Classification Metrics")
    print("=" * 70)

    # Load entry policy checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    embedding_dim = config.get("embedding_dim", 256)
    hidden_dim = config.get("hidden_dim", 256)
    num_layers = config.get("num_layers", 2)
    num_actions = config.get("num_actions", 5)

    print(f"\nEntry Policy config:")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Num actions: {num_actions}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    # Load LeJEPA
    print(f"\nLoading LeJEPA from: {lejepa_checkpoint}")
    lejepa, _ = LeJEPA.load_checkpoint(lejepa_checkpoint, device=str(device))
    lejepa = lejepa.to(device)
    lejepa.eval()
    for param in lejepa.parameters():
        param.requires_grad = False

    # Load entry policy
    entry_policy = EntryPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_actions=num_actions,
    )
    entry_policy.load_state_dict(checkpoint["policy_state_dict"])
    entry_policy = entry_policy.to(device)
    entry_policy.eval()

    print("\nLoading dataset...")
    train_dataset, val_dataset, _ = create_combined_dataset(
        stocks_dir=Path(data_dir),
        options_dir=Path(options_dir) if options_dir else None,
        start_date=start_date,
        end_date=end_date,
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  Val samples: {len(val_dataset)}")

    # Collect predictions
    print("\nRunning inference...")
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for context_batch, target_batch in tqdm(val_loader, desc="Evaluating"):
            context_batch = context_batch.to(device)

            # Get embeddings from LeJEPA
            embeddings = lejepa.context_encoder(context_batch, return_all_tokens=False)

            # Get predictions from entry policy
            output = entry_policy(embeddings.float())
            probs = output["action_probs"]
            preds = probs.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_probs = torch.cat(all_probs, dim=0)

    # Calculate metrics
    action_names = {
        0: "HOLD",
        1: "CALL_ATM" if num_actions == 5 else "CALL",
        2: "CALL_OTM" if num_actions == 5 else "PUT",
        3: "PUT_ATM" if num_actions == 5 else "N/A",
        4: "PUT_OTM" if num_actions == 5 else "N/A",
    }

    # Action distribution
    action_counts = {}
    total = len(all_preds)
    for action in range(num_actions):
        count = (all_preds == action).sum().item()
        action_counts[action_names[action]] = count
        action_counts[f"{action_names[action]}_pct"] = count / total * 100

    # Confidence statistics
    max_probs = all_probs.max(dim=-1).values
    avg_confidence = max_probs.mean().item()
    high_conf_count = (max_probs > 0.7).sum().item()

    # Signal rate (non-HOLD predictions)
    signal_count = (all_preds != 0).sum().item()
    signal_rate = signal_count / total

    metrics = {
        "total_samples": total,
        "signal_rate": signal_rate,
        "avg_confidence": avg_confidence,
        "high_confidence_rate": high_conf_count / total,
        **action_counts,
    }

    return metrics


# =============================================================================
# Exit Policy Evaluation (Simulated Trading)
# =============================================================================


def estimate_option_price(
    spy_price: float,
    position_type: str,
    time_to_close_hours: float,
    volatility: float = 0.15,
) -> float:
    """Estimate ATM option price."""
    T = time_to_close_hours / (252 * 6.5)
    price = 0.4 * spy_price * volatility * np.sqrt(max(T, 0.0001))
    return max(price, 0.01)


def compute_option_pnl(
    entry_spy_price: float,
    exit_spy_price: float,
    entry_option_price: float,
    position_type: str,
    entry_time_to_close: float,
    exit_time_to_close: float,
    position_value: float = 1000.0,
) -> Tuple[float, float]:
    """Compute P&L for an options position."""
    exit_option_price = estimate_option_price(exit_spy_price, position_type, exit_time_to_close)

    spy_move = exit_spy_price - entry_spy_price
    delta = 0.5

    if position_type == "call":
        intrinsic_change = max(spy_move * delta, -entry_option_price)
    else:
        intrinsic_change = max(-spy_move * delta, -entry_option_price)

    theta_decay = entry_option_price - exit_option_price
    if exit_time_to_close < entry_time_to_close:
        theta_decay = max(theta_decay, 0)
    else:
        theta_decay = 0

    option_price_change = intrinsic_change - theta_decay * 0.3

    num_contracts = position_value / (entry_option_price * 100)
    pnl_dollars = option_price_change * 100 * num_contracts
    pnl_pct = (option_price_change / entry_option_price) * 100 if entry_option_price > 0 else 0

    return pnl_dollars, pnl_pct


def evaluate_exit_policy(
    checkpoint_path: str,
    lejepa_checkpoint: str,
    entry_checkpoint: str,
    device: torch.device,
    data_dir: str = "data/stocks",
    options_dir: str = "data/options",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_days: int = 50,
) -> Dict[str, float]:
    """Evaluate exit policy with simulated trading."""
    print("=" * 70)
    print("Exit Policy Evaluation: Simulated Trading")
    print("=" * 70)

    # Load exit policy
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    embedding_dim = config.get("embedding_dim", 256)
    hidden_dim = config.get("hidden_dim", 256)
    num_layers = config.get("num_layers", 2)
    context_dim = config.get("context_dim", 4)

    print(f"\nExit Policy config:")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Context dim: {context_dim}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    # Load LeJEPA
    print(f"\nLoading LeJEPA from: {lejepa_checkpoint}")
    lejepa, _ = LeJEPA.load_checkpoint(lejepa_checkpoint, device=str(device))
    lejepa = lejepa.to(device)
    lejepa.eval()

    # Load entry policy
    print(f"Loading Entry Policy from: {entry_checkpoint}")
    entry_ckpt = torch.load(entry_checkpoint, map_location=device, weights_only=False)
    entry_config = entry_ckpt.get("config", {})
    entry_policy = EntryPolicy(
        embedding_dim=entry_config.get("embedding_dim", 256),
        hidden_dim=entry_config.get("hidden_dim", 256),
        num_layers=entry_config.get("num_layers", 2),
        num_actions=entry_config.get("num_actions", 5),
    )
    entry_policy.load_state_dict(entry_ckpt["policy_state_dict"])
    entry_policy = entry_policy.to(device)
    entry_policy.eval()

    # Load exit policy
    exit_policy = ExitPolicy(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        context_dim=context_dim,
    )
    exit_policy.load_state_dict(checkpoint["policy_state_dict"])
    exit_policy = exit_policy.to(device)
    exit_policy.eval()

    # Load data as days
    print("\nLoading data...")
    df = load_normalized_data(
        stocks_dir=data_dir,
        options_dir=options_dir,
        start_date=start_date,
        end_date=end_date,
    )

    df["date"] = df.index.date
    days = df["date"].unique()

    market_open = time(9, 30)
    market_close_time = time(16, 0)

    day_dfs = []
    for day in days:
        day_data = df[df["date"] == day].copy()
        day_data = day_data[
            (day_data.index.time >= market_open) &
            (day_data.index.time <= market_close_time)
        ]
        if len(day_data) >= 100:
            day_dfs.append(day_data.drop(columns=["date"]))

    # Use validation set (last 20%)
    n_val = max(1, int(len(day_dfs) * 0.2))
    val_days = day_dfs[-n_val:][:max_days]
    print(f"  Evaluating on {len(val_days)} days")

    # Create patcher
    base_cols = {"open", "high", "low", "close", "volume"}
    extra_columns = [col for col in df.columns if col not in base_cols and col != "date"]
    patcher = MarketPatch(patch_length=32, extra_columns=extra_columns)

    # Simulate trading
    total_pnl = 0.0
    total_trades = 0
    winning_trades = 0
    all_trade_pnls = []
    all_holding_bars = []
    exit_reasons = {"policy": 0, "eod": 0}

    for day_df in tqdm(val_days, desc="Simulating"):
        closes = day_df["close"].values
        timestamps = day_df.index
        position = None
        current_idx = 32

        while current_idx < len(day_df) - 1:
            # Get time to close
            ts = timestamps[current_idx]
            close_dt = datetime.combine(ts.date(), market_close_time)
            time_to_close = max((close_dt - ts).total_seconds() / 3600, 0)

            if time_to_close < 0.5:  # Last 30 min
                break

            # Create patch
            try:
                patch = patcher.create_patch(day_df, current_idx - 32)
            except (ValueError, IndexError):
                current_idx += 1
                continue

            with torch.no_grad():
                state_tensor = patch.unsqueeze(0).to(device)
                embedding = lejepa.context_encoder(state_tensor, return_all_tokens=False).float()

                if position is None:
                    # Try to open position
                    entry_output = entry_policy(embedding)
                    probs = entry_output["action_probs"][0]
                    action_idx = probs.argmax().item()

                    if action_idx != 0:  # Not HOLD
                        current_price = closes[current_idx]
                        position_type = "call" if action_idx in [1, 2] else "put"
                        option_price = estimate_option_price(current_price, position_type, time_to_close)
                        position = {
                            "type": position_type,
                            "entry_price": current_price,
                            "entry_idx": current_idx,
                            "option_price": option_price,
                            "entry_time_to_close": time_to_close,
                        }
                else:
                    # Check exit
                    current_price = closes[current_idx]
                    pnl, pnl_pct = compute_option_pnl(
                        position["entry_price"],
                        current_price,
                        position["option_price"],
                        position["type"],
                        position["entry_time_to_close"],
                        time_to_close,
                    )

                    # Build context
                    pos_type = 1.0 if position["type"] == "call" else -1.0
                    normalized_pnl = pnl / 1000.0
                    bars_held = (current_idx - position["entry_idx"]) / 360
                    normalized_time = time_to_close / 6.5

                    context = torch.tensor(
                        [[pos_type, normalized_pnl, bars_held, normalized_time]],
                        dtype=torch.float32,
                        device=device,
                    )

                    exit_output = exit_policy(embedding, context, return_value=False)
                    exit_action = exit_output["action_probs"][0].argmax().item()

                    if exit_action == 1:  # CLOSE
                        total_pnl += pnl
                        total_trades += 1
                        all_trade_pnls.append(pnl)
                        all_holding_bars.append(current_idx - position["entry_idx"])
                        if pnl > 0:
                            winning_trades += 1
                        exit_reasons["policy"] += 1
                        position = None

            current_idx += 1

        # Force close EOD
        if position is not None:
            current_price = closes[min(current_idx, len(closes) - 1)]
            pnl, _ = compute_option_pnl(
                position["entry_price"],
                current_price,
                position["option_price"],
                position["type"],
                position["entry_time_to_close"],
                0.1,
            )
            total_pnl += pnl
            total_trades += 1
            all_trade_pnls.append(pnl)
            all_holding_bars.append(current_idx - position["entry_idx"])
            if pnl > 0:
                winning_trades += 1
            exit_reasons["eod"] += 1

    metrics = {
        "total_pnl": total_pnl,
        "avg_daily_pnl": total_pnl / len(val_days) if val_days else 0,
        "total_trades": total_trades,
        "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
        "avg_trade_pnl": np.mean(all_trade_pnls) if all_trade_pnls else 0,
        "avg_holding_bars": np.mean(all_holding_bars) if all_holding_bars else 0,
        "policy_exits": exit_reasons["policy"],
        "eod_exits": exit_reasons["eod"],
    }

    return metrics


# =============================================================================
# Main
# =============================================================================


def print_metrics(metrics: Dict, model_type: str) -> None:
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS ({model_type.upper()})")
    print("=" * 70)

    if model_type == "lejepa":
        print(f"\nLinear Probe Results:")
        print(f"  Train Accuracy: {metrics.get('train_accuracy', 0):.2%}")
        print(f"  Val Accuracy:   {metrics.get('val_accuracy', 0):.2%}")
        print(f"  Val Positive Rate: {metrics.get('val_positive_rate', 0):.2%}")

        baseline = metrics.get('val_positive_rate', 0.5)
        edge = metrics.get('val_accuracy', 0.5) - max(baseline, 1 - baseline)
        print(f"\n  Random Baseline: {max(baseline, 1-baseline):.2%}")
        print(f"  Edge over baseline: {edge:+.2%}")

        if metrics.get('val_accuracy', 0.5) > 0.53:
            print("\n  PASS: Model shows predictive power (>53%)")
        elif metrics.get('val_accuracy', 0.5) > 0.51:
            print("\n  MARGINAL: Slight edge (51-53%)")
        else:
            print("\n  BASELINE: No better than random (~50%)")

    elif model_type == "entry":
        print(f"\nClassification Results:")
        print(f"  Total Samples: {metrics.get('total_samples', 0):,}")
        print(f"  Signal Rate: {metrics.get('signal_rate', 0):.2%}")
        print(f"  Avg Confidence: {metrics.get('avg_confidence', 0):.2%}")
        print(f"  High Confidence (>70%): {metrics.get('high_confidence_rate', 0):.2%}")

        print(f"\n  Action Distribution:")
        for key, value in metrics.items():
            if "_pct" in key:
                name = key.replace("_pct", "")
                count = metrics.get(name, 0)
                print(f"    {name}: {count:,} ({value:.1f}%)")

    elif model_type == "exit":
        print(f"\nTrading Simulation Results:")
        print(f"  Total P&L: ${metrics.get('total_pnl', 0):+,.2f}")
        print(f"  Avg Daily P&L: ${metrics.get('avg_daily_pnl', 0):+,.2f}")
        print(f"  Total Trades: {metrics.get('total_trades', 0)}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"  Avg Trade P&L: ${metrics.get('avg_trade_pnl', 0):+.2f}")
        print(f"  Avg Holding Bars: {metrics.get('avg_holding_bars', 0):.1f}")

        print(f"\n  Exit Reasons:")
        print(f"    Policy: {metrics.get('policy_exits', 0)}")
        print(f"    EOD: {metrics.get('eod_exits', 0)}")

    print("\n" + "=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Model Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["lejepa", "entry", "exit"],
        default=None,
        help="Force model type (auto-detected if not specified)",
    )
    parser.add_argument(
        "--lejepa-checkpoint",
        type=str,
        default=None,
        help="LeJEPA checkpoint (required for entry/exit evaluation)",
    )
    parser.add_argument(
        "--entry-checkpoint",
        type=str,
        default=None,
        help="Entry policy checkpoint (required for exit evaluation)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/stocks",
        help="Stocks data directory",
    )
    parser.add_argument(
        "--options-dir",
        type=str,
        default="data/options",
        help="Options data directory",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--probe-epochs",
        type=int,
        default=100,
        help="Epochs for linear probe training (LeJEPA only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cuda', or 'cpu'",
    )

    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\nDevice: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load checkpoint and detect type
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.model_type:
        model_type = args.model_type
        print(f"Model type (forced): {model_type}")
    else:
        info = detect_model_type(checkpoint)
        model_type = info.model_type
        print(f"Model type (auto-detected): {model_type}")

    # Run appropriate evaluation
    if model_type == "lejepa":
        metrics = evaluate_lejepa(
            checkpoint_path=args.checkpoint,
            device=device,
            data_dir=args.data_dir,
            options_dir=args.options_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            batch_size=args.batch_size,
            probe_epochs=args.probe_epochs,
        )

    elif model_type == "entry":
        if not args.lejepa_checkpoint:
            # Try to find lejepa checkpoint in same directory
            ckpt_dir = Path(args.checkpoint).parent.parent
            lejepa_path = ckpt_dir / "lejepa" / "lejepa_best.pt"
            if not lejepa_path.exists():
                print("\nError: --lejepa-checkpoint required for entry policy evaluation")
                return
            args.lejepa_checkpoint = str(lejepa_path)

        metrics = evaluate_entry_policy(
            checkpoint_path=args.checkpoint,
            lejepa_checkpoint=args.lejepa_checkpoint,
            device=device,
            data_dir=args.data_dir,
            options_dir=args.options_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            batch_size=args.batch_size,
        )

    elif model_type == "exit":
        if not args.lejepa_checkpoint or not args.entry_checkpoint:
            print("\nError: --lejepa-checkpoint and --entry-checkpoint required for exit evaluation")
            return

        metrics = evaluate_exit_policy(
            checkpoint_path=args.checkpoint,
            lejepa_checkpoint=args.lejepa_checkpoint,
            entry_checkpoint=args.entry_checkpoint,
            device=device,
            data_dir=args.data_dir,
            options_dir=args.options_dir,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    else:
        print(f"Unknown model type: {model_type}")
        return

    # Print results
    print_metrics(metrics, model_type)


if __name__ == "__main__":
    main()
