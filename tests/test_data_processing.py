"""
Unit tests for data processing components.
Tests MarketPatch and InMemoryDataset with synthetic/pre-normalized data.

NOTE: MarketPatch now expects PRE-NORMALIZED DataFrames from the loader.
All normalization happens in normalize_features() during cache building.
"""
import pytest
import torch
import numpy as np
import pandas as pd

from src.data.synthetic import SyntheticMarketData
from src.data.processing import MarketPatch, InMemoryDataset, create_dataloader


def create_normalized_dataframe(num_bars: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Create a sample pre-normalized DataFrame for testing MarketPatch.

    This simulates data that has already been processed by normalize_features().
    """
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02 14:30", periods=num_bars, freq="1min")

    # Normalized OHLC (log-returns, centered around 0, small variance)
    df = pd.DataFrame({
        "open": np.random.randn(num_bars) * 0.01,
        "high": np.random.randn(num_bars) * 0.01,
        "low": np.random.randn(num_bars) * 0.01,
        "close": np.random.randn(num_bars) * 0.01,
        # Normalized volume (centered around 0)
        "volume": np.random.randn(num_bars) * 0.5,
        # Time features (already normalized)
        "time_to_close": np.linspace(1.0, 0.0, num_bars),  # [0, 1]
        "sin_time": np.sin(np.linspace(0, 2*np.pi, num_bars)),  # [-1, 1]
        "cos_time": np.cos(np.linspace(0, 2*np.pi, num_bars)),  # [-1, 1]
    }, index=dates)

    return df


class TestMarketPatch:
    """Test MarketPatch class with pre-normalized data."""

    @pytest.fixture
    def normalized_data(self) -> pd.DataFrame:
        """Generate sample pre-normalized market data."""
        return create_normalized_dataframe(num_bars=100)

    def test_initialization(self) -> None:
        """Test MarketPatch initialization."""
        patcher = MarketPatch(patch_length=32)
        assert patcher.patch_length == 32

        patcher_with_extra = MarketPatch(patch_length=32, extra_columns=["implied_volatility", "net_gex"])
        assert patcher.patch_length == 32

    def test_create_patch_basic(self, normalized_data: pd.DataFrame) -> None:
        """Test basic patch creation from pre-normalized data."""
        patcher = MarketPatch(patch_length=32)

        patch = patcher.create_patch(normalized_data, start_idx=0)

        assert isinstance(patch, torch.Tensor)
        assert patch.shape[0] == 32  # patch_length
        assert patch.shape[1] == 8  # number of columns in normalized data
        assert patch.dtype == torch.float32
        assert not torch.isnan(patch).any()
        assert not torch.isinf(patch).any()

    def test_create_patch_extracts_correct_window(self, normalized_data: pd.DataFrame) -> None:
        """Test that create_patch extracts the correct window."""
        patcher = MarketPatch(patch_length=32)

        # Get patch from index 10
        patch = patcher.create_patch(normalized_data, start_idx=10)

        # First element of patch should match row 10 of data
        expected_first = normalized_data.iloc[10][["open", "high", "low", "close", "volume",
                                                    "time_to_close", "sin_time", "cos_time"]].values
        actual_first = patch[0].numpy()

        np.testing.assert_array_almost_equal(actual_first, expected_first, decimal=5)

    def test_create_patch_with_extra_columns(self) -> None:
        """Test patch creation with extra columns."""
        # Create data with additional features
        df = create_normalized_dataframe(num_bars=100)
        df["implied_volatility"] = np.random.randn(100) * 0.1  # Normalized IV
        df["net_gex"] = np.random.randn(100) * 0.1  # Normalized GEX

        patcher = MarketPatch(patch_length=32, extra_columns=["implied_volatility", "net_gex"])
        patch = patcher.create_patch(df, start_idx=0)

        assert patch.shape == (32, 10)  # 8 base + 2 extra
        assert patcher.feature_dim == 10

    def test_create_patch_insufficient_data(self, normalized_data: pd.DataFrame) -> None:
        """Test error handling with insufficient data."""
        patcher = MarketPatch(patch_length=32)

        # Should raise error when not enough data
        with pytest.raises(ValueError, match="Insufficient data"):
            patcher.create_patch(normalized_data, start_idx=90)

    def test_create_patch_partial(self, normalized_data: pd.DataFrame) -> None:
        """Test partial patch creation with padding."""
        patcher = MarketPatch(patch_length=32)

        # Create partial patch with only 10 actual candles
        patch = patcher.create_patch(normalized_data, start_idx=0, actual_length=10)

        assert patch.shape == (32, 8)  # Still full patch_length

        # First 22 rows should be padding (zeros)
        assert torch.allclose(patch[:22], torch.zeros(22, 8))

        # Last 10 rows should be actual data
        assert not torch.allclose(patch[22:], torch.zeros(10, 8))

    def test_create_patch_clips_extreme_values(self) -> None:
        """Test that extreme values are clipped."""
        df = create_normalized_dataframe(num_bars=100)
        # Add extreme values
        df.loc[df.index[0], "close"] = 100.0  # Very large
        df.loc[df.index[1], "close"] = -100.0  # Very negative

        patcher = MarketPatch(patch_length=32)
        patch = patcher.create_patch(df, start_idx=0)

        # Values should be clipped to [-10, 10]
        assert patch.min() >= -10.0
        assert patch.max() <= 10.0

    def test_create_patch_handles_nan(self) -> None:
        """Test that NaN values are replaced with 0."""
        df = create_normalized_dataframe(num_bars=100)
        df.loc[df.index[0], "close"] = np.nan

        patcher = MarketPatch(patch_length=32)
        patch = patcher.create_patch(df, start_idx=0)

        # Should not contain NaN
        assert not torch.isnan(patch).any()


class TestInMemoryDataset:
    """Test InMemoryDataset class."""

    @pytest.fixture
    def sample_dataset(self) -> InMemoryDataset:
        """Create sample dataset from pre-normalized data."""
        df = create_normalized_dataframe(num_bars=500)  # Larger dataset for batching

        dataset = InMemoryDataset.from_dataframe(
            df,
            patch_length=32,
            stride=8,  # Smaller stride for more samples
            context_horizon=4,
            target_horizon=8,
        )
        return dataset

    def test_dataset_creation(self, sample_dataset: InMemoryDataset) -> None:
        """Test dataset creation."""
        assert len(sample_dataset) > 0
        assert len(sample_dataset.patches) == len(sample_dataset)

    def test_dataset_getitem(self, sample_dataset: InMemoryDataset) -> None:
        """Test dataset __getitem__."""
        patch, label = sample_dataset[0]

        assert isinstance(patch, torch.Tensor)
        assert patch.shape[0] == 32  # patch_length
        assert patch.dtype == torch.float32

    def test_dataloader_creation(self, sample_dataset: InMemoryDataset) -> None:
        """Test dataloader creation."""
        loader = create_dataloader(
            sample_dataset,
            batch_size=16,
            shuffle=True,
            drop_last=True,
        )

        batch = next(iter(loader))
        context, target = batch

        assert context.shape[0] == 16  # batch_size
        assert context.shape[1] == 32  # patch_length
        assert target.shape[0] == 16
        assert target.shape[1] == 32

    def test_large_batch_support(self) -> None:
        """Test that large batches work (SIGReg requires 2048+)."""
        df = create_normalized_dataframe(num_bars=10000)  # More data for large batches

        dataset = InMemoryDataset.from_dataframe(
            df,
            patch_length=32,
            stride=2,  # Very small stride for maximum samples
            context_horizon=4,
            target_horizon=8,
        )

        # Should have enough samples for large batches
        assert len(dataset) > 2048

        loader = create_dataloader(
            dataset,
            batch_size=2048,
            shuffle=True,
            drop_last=True,
        )

        batch = next(iter(loader))
        context, target = batch

        assert context.shape[0] == 2048


def test_end_to_end_pipeline() -> None:
    """Test complete pipeline from data to patches."""
    # Create pre-normalized data
    df = create_normalized_dataframe(num_bars=500)

    # Create dataset
    dataset = InMemoryDataset.from_dataframe(
        df,
        patch_length=32,
        stride=8,
        context_horizon=4,
        target_horizon=8,
    )

    # Create dataloader
    loader = create_dataloader(dataset, batch_size=64, shuffle=True)

    # Iterate through one batch
    for context, target in loader:
        # Verify batch shapes
        assert context.ndim == 3  # [batch, time, features]
        assert target.ndim == 3
        assert context.shape[0] == 64
        assert context.shape[1] == 32

        # Verify no NaN or Inf
        assert not torch.isnan(context).any()
        assert not torch.isinf(context).any()
        break  # Only check first batch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
