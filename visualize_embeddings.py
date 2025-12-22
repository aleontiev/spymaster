#!/usr/bin/env python3
"""
Embedding Visualization for LeJEPA.

Provides tools to visualize learned embeddings and detect representation collapse:
- PCA projection to 2D/3D
- Embedding covariance heatmap
- Dimension variance analysis
- Training dynamics tracking

Usage:
    # Using direct path:
    uv run python visualize_embeddings.py --checkpoint checkpoints/lejepa_v3/lejepa_best.pt

    # Using checkpoint ID (name from registry):
    uv run python visualize_embeddings.py --checkpoint lejepa-2021-01-02-to-2025-09-01
"""
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.processing import InMemoryDataset, create_dataloader
from src.data.synthetic import create_test_dataset
from src.model.lejepa import LeJEPA
from src.managers.checkpoint_manager import resolve_checkpoint_file, get_manager


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize LeJEPA embeddings")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint name (from registry) or path to .pt file",
    )
    parser.add_argument(
        "--num_days",
        type=int,
        default=10,
        help="Number of synthetic days for visualization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Output directory for plots",
    )
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

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Determine device based on argument and availability."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


@torch.no_grad()
def extract_embeddings(
    model: LeJEPA,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract embeddings from model.

    Args:
        model: LeJEPA model
        dataloader: Data loader
        device: Device

    Returns:
        Tuple of (context_embeddings, target_embeddings, predicted_embeddings)
    """
    model.eval()

    context_embs = []
    target_embs = []
    predicted_embs = []

    for context, target in dataloader:
        context = context.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            output = model(context, target, return_loss=False)

        context_embs.append(output["context_embedding"].float().cpu().numpy())
        target_embs.append(output["target_embedding"].float().cpu().numpy())
        predicted_embs.append(output["predicted_embedding"].float().cpu().numpy())

    return (
        np.concatenate(context_embs, axis=0),
        np.concatenate(target_embs, axis=0),
        np.concatenate(predicted_embs, axis=0),
    )


def compute_pca(
    embeddings: np.ndarray, n_components: int = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA projection of embeddings.

    Args:
        embeddings: Embedding matrix [N, D]
        n_components: Number of PCA components

    Returns:
        Tuple of (projected, explained_variance_ratio, singular_values)
    """
    # Center the data
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    # SVD for PCA
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    # Project to n_components
    projected = U[:, :n_components] * S[:n_components]

    # Explained variance ratio
    total_var = (S ** 2).sum()
    explained_var_ratio = (S ** 2) / total_var

    return projected, explained_var_ratio, S


def analyze_embeddings(embeddings: np.ndarray) -> dict:
    """
    Compute embedding statistics for collapse detection.

    Args:
        embeddings: Embedding matrix [N, D]

    Returns:
        Dictionary of statistics
    """
    stats = {}

    # Basic statistics
    stats["mean"] = embeddings.mean()
    stats["std"] = embeddings.std()
    stats["min"] = embeddings.min()
    stats["max"] = embeddings.max()

    # Per-sample norms
    norms = np.linalg.norm(embeddings, axis=1)
    stats["norm_mean"] = norms.mean()
    stats["norm_std"] = norms.std()

    # Per-dimension variance (collapse indicator)
    dim_var = embeddings.var(axis=0)
    stats["dim_var_mean"] = dim_var.mean()
    stats["dim_var_std"] = dim_var.std()
    stats["dim_var_min"] = dim_var.min()
    stats["dim_var_max"] = dim_var.max()

    # Effective rank (measure of dimensionality usage)
    # Higher is better - means more dimensions are being used
    _, S, _ = np.linalg.svd(embeddings - embeddings.mean(axis=0), full_matrices=False)
    normalized_S = S / S.sum()
    entropy = -np.sum(normalized_S * np.log(normalized_S + 1e-10))
    stats["effective_rank"] = np.exp(entropy)

    # Covariance deviation from identity (SIGReg objective)
    centered = embeddings - embeddings.mean(axis=0)
    cov = (centered.T @ centered) / len(embeddings)
    identity = np.eye(cov.shape[0])
    stats["cov_frobenius_norm"] = np.linalg.norm(cov - identity, "fro")

    return stats


def print_embedding_analysis(
    context_embs: np.ndarray,
    target_embs: np.ndarray,
    predicted_embs: np.ndarray,
) -> None:
    """Print detailed embedding analysis."""
    print("\n" + "=" * 70)
    print("Embedding Analysis")
    print("=" * 70)

    for name, embs in [
        ("Context", context_embs),
        ("Target", target_embs),
        ("Predicted", predicted_embs),
    ]:
        stats = analyze_embeddings(embs)
        print(f"\n{name} Embeddings:")
        print(f"  Shape: {embs.shape}")
        print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Norm - Mean: {stats['norm_mean']:.4f}, Std: {stats['norm_std']:.4f}")
        print(f"  Dim Variance - Mean: {stats['dim_var_mean']:.4f}, "
              f"Std: {stats['dim_var_std']:.4f}")
        print(f"  Dim Variance - Range: [{stats['dim_var_min']:.4f}, "
              f"{stats['dim_var_max']:.4f}]")
        print(f"  Effective Rank: {stats['effective_rank']:.2f} / {embs.shape[1]}")
        print(f"  Cov-Identity Frobenius: {stats['cov_frobenius_norm']:.4f}")

    # Prediction quality
    pred_error = np.mean((predicted_embs - target_embs) ** 2)
    pred_cosine = np.mean(
        np.sum(predicted_embs * target_embs, axis=1)
        / (np.linalg.norm(predicted_embs, axis=1) * np.linalg.norm(target_embs, axis=1))
    )
    print(f"\nPrediction Quality:")
    print(f"  MSE: {pred_error:.4f}")
    print(f"  Cosine Similarity: {pred_cosine:.4f}")

    # Collapse detection
    print("\n" + "-" * 70)
    print("Collapse Detection:")
    context_stats = analyze_embeddings(context_embs)

    if context_stats["effective_rank"] < embs.shape[1] * 0.1:
        print("  WARNING: Low effective rank - possible dimensional collapse!")
    else:
        print(f"  OK: Effective rank is {context_stats['effective_rank']:.1f} "
              f"({100*context_stats['effective_rank']/embs.shape[1]:.1f}% of dimensions)")

    if context_stats["dim_var_std"] / context_stats["dim_var_mean"] > 2.0:
        print("  WARNING: High variance in dimension variances - uneven usage!")
    else:
        print("  OK: Dimensions are being used relatively evenly")

    if context_stats["std"] < 0.1:
        print("  WARNING: Very low embedding variance - possible complete collapse!")
    else:
        print(f"  OK: Embedding std is {context_stats['std']:.4f}")


def save_pca_plot(
    embeddings: np.ndarray,
    output_path: Path,
    title: str = "PCA of Embeddings",
) -> None:
    """
    Save PCA visualization to file (text-based for environments without matplotlib).

    Args:
        embeddings: Embedding matrix [N, D]
        output_path: Output file path
        title: Plot title
    """
    projected, explained_var, singular_values = compute_pca(embeddings, n_components=2)

    # Save as text report
    with open(output_path.with_suffix(".txt"), "w") as f:
        f.write(f"{title}\n")
        f.write("=" * 70 + "\n\n")

        f.write("PCA Summary:\n")
        f.write(f"  Total samples: {len(embeddings)}\n")
        f.write(f"  Embedding dim: {embeddings.shape[1]}\n")
        f.write(f"  PC1 explained variance: {explained_var[0]*100:.2f}%\n")
        f.write(f"  PC2 explained variance: {explained_var[1]*100:.2f}%\n")
        f.write(f"  Top 10 components explain: {explained_var[:10].sum()*100:.2f}%\n")
        f.write(f"  Top 50 components explain: {explained_var[:50].sum()*100:.2f}%\n\n")

        f.write("Singular Value Spectrum (first 20):\n")
        for i, s in enumerate(singular_values[:20]):
            bar_len = int(s / singular_values[0] * 40)
            f.write(f"  {i+1:3d}: {s:8.2f} {'#' * bar_len}\n")

        f.write("\nPC1 vs PC2 Statistics:\n")
        f.write(f"  PC1 range: [{projected[:, 0].min():.2f}, {projected[:, 0].max():.2f}]\n")
        f.write(f"  PC2 range: [{projected[:, 1].min():.2f}, {projected[:, 1].max():.2f}]\n")
        f.write(f"  PC1 std: {projected[:, 0].std():.2f}\n")
        f.write(f"  PC2 std: {projected[:, 1].std():.2f}\n")

    print(f"  Saved PCA analysis to {output_path.with_suffix('.txt')}")

    # Try to save plot if matplotlib is available
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PCA scatter
        ax1 = axes[0]
        scatter = ax1.scatter(
            projected[:, 0],
            projected[:, 1],
            c=np.arange(len(projected)),
            cmap="viridis",
            alpha=0.5,
            s=10,
        )
        ax1.set_xlabel(f"PC1 ({explained_var[0]*100:.1f}%)")
        ax1.set_ylabel(f"PC2 ({explained_var[1]*100:.1f}%)")
        ax1.set_title(title)
        plt.colorbar(scatter, ax=ax1, label="Sample Index")

        # Singular value spectrum
        ax2 = axes[1]
        ax2.semilogy(singular_values[:50], "b-", linewidth=2)
        ax2.set_xlabel("Component")
        ax2.set_ylabel("Singular Value (log)")
        ax2.set_title("Singular Value Spectrum")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path.with_suffix(".png"), dpi=150)
        plt.close()

        print(f"  Saved PCA plot to {output_path.with_suffix('.png')}")

    except ImportError:
        print("  (matplotlib not available - skipping image plot)")


def main() -> None:
    """Main visualization function."""
    args = parse_args()

    print("=" * 70)
    print("LeJEPA Embedding Visualization")
    print("=" * 70)

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    device = get_device(args.device)
    print(f"\nDevice: {device}")

    # Resolve checkpoint (supports name from registry or direct path)
    checkpoint_path = resolve_checkpoint_file(args.checkpoint)

    if checkpoint_path is None:
        # Try direct path as fallback
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"\nError: Checkpoint not found: {args.checkpoint}")
            print("\nAvailable checkpoints in registry:")
            mgr = get_manager()
            for cp in mgr.list(model_type="lejepa"):
                print(f"  - {cp['name']} ({cp['status']})")
            return

    print(f"\nLoading checkpoint: {checkpoint_path}")

    # Load model
    model, checkpoint = LeJEPA.load_checkpoint(str(checkpoint_path), device=str(device))
    model = model.to(device)

    epoch = checkpoint.get("epoch", "unknown")
    print(f"  Epoch: {epoch}")

    # Load from training cache to ensure feature consistency (24 features)
    import torch as th

    print("\nLoading data from training cache...")
    # Use latest training cache
    cache_file = Path("data/cache/combined_gex_dataset_45ce21b9a324.pt")
    if not cache_file.exists():
        # Fallback: find any combined cache
        import glob
        caches = sorted(glob.glob("data/cache/combined_gex_dataset_*.pt"))
        if caches:
            cache_file = Path(caches[-1])
        else:
            print("ERROR: No training cache found. Run training first.")
            return

    print(f"  Cache file: {cache_file}")
    cache_data = th.load(cache_file, weights_only=False)

    # Use validation patches (smaller, representative sample)
    val_patches = cache_data["val_patches"][:args.num_days * 300]  # ~300 patches/day
    val_labels = cache_data["val_labels"][:args.num_days * 300]

    print(f"  Loaded {len(val_patches)} patches with {val_patches[0].shape[-1]} features")

    # Create dataset directly from patches
    dataset = InMemoryDataset(patches=val_patches, labels=val_labels, device="cpu")

    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"  Samples: {len(dataset)}")
    print(f"  Batches: {len(dataloader)}")

    # Extract embeddings
    print("\nExtracting embeddings...")
    context_embs, target_embs, predicted_embs = extract_embeddings(
        model, dataloader, device
    )

    # Analyze
    print_embedding_analysis(context_embs, target_embs, predicted_embs)

    # Save visualizations
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Saving Visualizations")
    print("=" * 70)

    save_pca_plot(
        context_embs,
        output_dir / f"pca_context_epoch{epoch}",
        title=f"Context Embeddings PCA (Epoch {epoch})",
    )

    save_pca_plot(
        predicted_embs,
        output_dir / f"pca_predicted_epoch{epoch}",
        title=f"Predicted Embeddings PCA (Epoch {epoch})",
    )

    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
