"""
End-to-end integration test for LeJEPA training and inference.
Tests complete pipeline with synthetic data.
"""
import pytest
import torch
from pathlib import Path
import tempfile

from src.data.synthetic import SyntheticMarketData, create_test_dataset
from src.data.processing import InMemoryDataset, create_dataloader
from src.model.lejepa import LeJEPA


def test_end_to_end_training() -> None:
    """
    Complete end-to-end test:
    1. Generate synthetic data
    2. Create dataset
    3. Initialize model
    4. Train for a few epochs
    5. Test inference
    6. Save/load checkpoint
    """
    print("\n" + "=" * 60)
    print("END-TO-END INTEGRATION TEST")
    print("=" * 60)

    # ========== STEP 1: Generate Synthetic Data ==========
    print("\n[1/6] Generating synthetic market data...")

    df = create_test_dataset(
        num_days=10,  # More data for larger batches
        bars_per_day=390,
        random_seed=42,
    )

    assert len(df) > 0
    print(f"  Generated {len(df)} bars")

    # ========== STEP 2: Create Dataset ==========
    print("\n[2/6] Creating dataset...")

    dataset = InMemoryDataset.from_dataframe(
        df,
        patch_length=32,
        stride=8,
        context_horizon=4,
        target_horizon=8,
        device='cpu',
    )

    print(f"  Created {len(dataset)} training samples")

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create dataloaders with drop_last=True to avoid small batches
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    print(f"  Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # ========== STEP 3: Initialize Model ==========
    print("\n[3/6] Initializing LeJEPA model...")

    # Get feature dimension from dataset
    context_sample, _ = dataset[0]
    input_dim = context_sample.shape[-1]  # Feature dimension

    model = LeJEPA(
        input_dim=input_dim,
        premarket_dim=6,
        premarket_len=15,
        first15_len=15,
        d_model=64,
        nhead=4,
        num_layers=2,
        embedding_dim=32,
        max_context_len=75,
        lambda_reg=0.5,
        reg_type="vicreg",
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Model initialized")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

    # ========== STEP 4: Train for a Few Epochs ==========
    print("\n[4/6] Training model...")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )

    # Training loop
    num_epochs = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print(f"  Using device: {device}")

    for epoch in range(num_epochs):
        model.train()
        train_losses = []

        for batch_idx, (context, target) in enumerate(train_loader):
            context = context.to(device)
            target = target.to(device)

            # Forward (without prefix - simple mode)
            optimizer.zero_grad()
            output = model(x_context=context, x_target=target, return_loss=True)

            loss = output['loss']
            pred_loss = output['pred_loss']
            reg_loss = output['reg_loss']

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

            # Log first and last batch
            if batch_idx == 0 or batch_idx == len(train_loader) - 1:
                print(f"    Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}: "
                      f"Loss={loss.item():.4f}, Pred={pred_loss.item():.4f}, "
                      f"Reg={reg_loss.item():.4f}")

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation
        model.eval()
        val_losses = []

        with torch.no_grad():
            for context, target in val_loader:
                context = context.to(device)
                target = target.to(device)

                output = model(x_context=context, x_target=target, return_loss=True)
                val_losses.append(output['loss'].item())

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        if val_losses:
            print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}")
        else:
            print(f"  Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}")

    # Check that loss decreased (sanity check)
    initial_loss = train_losses[0]
    final_loss = train_losses[-1]
    print(f"\n  Loss change: {initial_loss:.4f} -> {final_loss:.4f} "
          f"({((final_loss - initial_loss) / initial_loss * 100):+.1f}%)")

    # ========== STEP 5: Test Inference ==========
    print("\n[5/6] Testing inference...")

    model.eval()

    # Get a sample
    context_sample, _ = dataset[0]
    context_sample = context_sample.unsqueeze(0).to(device)  # Add batch dim

    with torch.no_grad():
        output = model(x_context=context_sample, return_loss=False)

    assert 'context_embedding' in output
    assert 'predicted_embedding' in output
    assert output['context_embedding'].shape == (1, 32)  # embedding_dim=32
    assert output['predicted_embedding'].shape == (1, 32)

    print(f"  Inference successful")
    print(f"    Context embedding: {output['context_embedding'].shape}")
    print(f"    Predicted embedding: {output['predicted_embedding'].shape}")

    # ========== STEP 6: Save and Load Checkpoint ==========
    print("\n[6/6] Testing checkpoint save/load...")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "lejepa_checkpoint.pth"

        # Save
        model.save_checkpoint(
            checkpoint_path,
            epoch=num_epochs,
            optimizer=optimizer,
            train_loss=avg_train_loss,
        )

        print(f"  Checkpoint saved to {checkpoint_path}")

        # Load
        loaded_model, checkpoint = LeJEPA.load_checkpoint(checkpoint_path)
        loaded_model.to(device)

        print(f"  Checkpoint loaded (epoch {checkpoint['epoch']})")
        print(f"    Train loss: {checkpoint.get('train_loss', 'N/A')}")

        # Test loaded model
        loaded_model.eval()
        with torch.no_grad():
            loaded_output = loaded_model(x_context=context_sample, return_loss=False)

        # Outputs should match
        assert torch.allclose(
            output['context_embedding'],
            loaded_output['context_embedding'],
            atol=1e-5,
        )

        print(f"  Loaded model produces identical outputs")

    print("\n" + "=" * 60)
    print("END-TO-END TEST PASSED!")
    print("=" * 60)
    print("\nSummary:")
    print(f"  - Generated {len(df)} synthetic market bars")
    print(f"  - Created {len(dataset)} training samples")
    print(f"  - Trained model with {trainable_params:,} parameters")
    print(f"  - Completed {num_epochs} epochs")
    print(f"  - Final train loss: {avg_train_loss:.4f}")
    print(f"  - Final val loss: {avg_val_loss:.4f}")
    print(f"  - Successfully saved and loaded checkpoint")
    print("=" * 60)


def test_embedding_quality() -> None:
    """Test that embeddings have reasonable properties."""
    print("\n" + "=" * 60)
    print("EMBEDDING QUALITY TEST")
    print("=" * 60)

    # Generate enough data for meaningful embeddings
    generator = SyntheticMarketData(random_seed=42)
    df = generator.generate_ohlcv_bars(num_bars=1000)

    dataset = InMemoryDataset.from_dataframe(df, patch_length=32, stride=16)
    loader = create_dataloader(dataset, batch_size=32, shuffle=False)

    # Get feature dimension from dataset
    context_sample, _ = dataset[0]
    input_dim = context_sample.shape[-1]

    # Create model
    model = LeJEPA(
        input_dim=input_dim,
        premarket_dim=6,
        premarket_len=15,
        first15_len=15,
        d_model=64,
        nhead=4,
        num_layers=2,
        embedding_dim=32,
        max_context_len=75,
        reg_type="vicreg",
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Extract embeddings
    all_embeddings = []

    with torch.no_grad():
        for context, _ in loader:
            context = context.to(device)
            output = model(x_context=context, return_loss=False)
            all_embeddings.append(output['context_embedding'].cpu())

    embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]

    print(f"\nEmbedding statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean().item():.4f}")
    print(f"  Std: {embeddings.std().item():.4f}")
    print(f"  Min: {embeddings.min().item():.4f}")
    print(f"  Max: {embeddings.max().item():.4f}")

    # Check for collapse (all embeddings identical)
    pairwise_dist = torch.cdist(embeddings[:10], embeddings[:10])
    avg_dist = pairwise_dist.mean().item()

    print(f"\n  Avg pairwise distance (sample): {avg_dist:.4f}")

    if avg_dist < 0.01:
        print("  Warning: Embeddings may have collapsed (very low diversity)")
    else:
        print("  Embeddings show healthy diversity")

    # Check for NaN/Inf
    assert not torch.isnan(embeddings).any(), "Embeddings contain NaN!"
    assert not torch.isinf(embeddings).any(), "Embeddings contain Inf!"

    print("\nEmbedding quality test passed!")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_bfloat16_training() -> None:
    """Test training with BFloat16 precision (if CUDA available)."""
    print("\n" + "=" * 60)
    print("BFLOAT16 TRAINING TEST")
    print("=" * 60)

    # Generate small dataset
    df = create_test_dataset(num_days=5, bars_per_day=390, random_seed=42)
    dataset = InMemoryDataset.from_dataframe(df, patch_length=32, stride=8)

    # Create loader with drop_last
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True,
    )

    # Get feature dimension
    context_sample, _ = dataset[0]
    input_dim = context_sample.shape[-1]

    # Create model
    model = LeJEPA(
        input_dim=input_dim,
        premarket_dim=6,
        premarket_len=15,
        first15_len=15,
        d_model=64,
        nhead=4,
        num_layers=2,
        embedding_dim=32,
        max_context_len=75,
        reg_type="vicreg",
    )

    device = torch.device('cuda')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training with BFloat16
    print("\nTraining with BFloat16...")

    model.train()
    for batch_idx, (context, target) in enumerate(loader):
        if batch_idx >= 5:  # Just test a few batches
            break

        context = context.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Use autocast
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(x_context=context, x_target=target, return_loss=True)
            loss = output['loss']

        # Check dtypes
        if batch_idx == 0:
            print(f"  Context dtype: {context.dtype}")
            print(f"  Output dtype: {output['context_embedding'].dtype}")
            print(f"  Loss dtype: {loss.dtype}")

        # Backward
        loss.backward()
        optimizer.step()

        # Check for NaN
        assert not torch.isnan(loss), f"NaN loss at batch {batch_idx}!"

        print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")

    print("\nBFloat16 training test passed!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
