"""
Unit tests for LeJEPA model components.
Tests the 3-part architecture with:
- Premarket prefix (15 basic candles, 6 features)
- First 15 minutes anchor (15 candles, 29 features) - always included
- Main context (0-75 candles, 29 features) - variable, can have gaps
- Checkpoint save/load
"""
import pytest
import torch
import tempfile
from pathlib import Path

from src.model.loss import PredictionLoss, SIGRegLoss, VICRegLoss, LeJEPALoss
from src.model.lejepa import LeJEPA


class TestLossFunctions:
    """Test loss function components."""

    def test_prediction_loss_mse(self) -> None:
        """Test prediction loss with MSE."""
        loss_fn = PredictionLoss(reduction="mean")

        predicted = torch.randn(64, 64)
        target = torch.randn(64, 64)

        loss = loss_fn(predicted, target)

        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_prediction_loss_identical_inputs(self) -> None:
        """Test prediction loss is zero for identical inputs."""
        loss_fn = PredictionLoss(reduction="mean")

        x = torch.randn(64, 64)
        loss = loss_fn(x, x)

        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)

    def test_sigreg_loss(self) -> None:
        """Test SIGReg loss computation."""
        loss_fn = SIGRegLoss(embedding_dim=64)

        # Need batch >= embedding_dim
        embeddings = torch.randn(128, 64)

        loss = loss_fn(embeddings)

        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_sigreg_isotropic_gaussian(self) -> None:
        """Test SIGReg loss is low for isotropic Gaussian."""
        loss_fn = SIGRegLoss(embedding_dim=64)

        # Large batch for accurate covariance estimation
        embeddings = torch.randn(512, 64)

        loss = loss_fn(embeddings)

        # Should be relatively small for Gaussian samples
        assert loss.item() < 100.0

    def test_vicreg_loss(self) -> None:
        """Test VICReg loss computation."""
        loss_fn = VICRegLoss(lambda_var=1.0, lambda_cov=10.0)

        embeddings = torch.randn(64, 64)

        loss = loss_fn(embeddings)

        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() >= 0

    def test_lejepa_loss(self) -> None:
        """Test combined LeJEPA loss."""
        loss_fn = LeJEPALoss(
            embedding_dim=64,
            lambda_reg=0.5,
        )

        # Batch must be >= embedding_dim for SIGReg
        predicted = torch.randn(128, 64)
        target = torch.randn(128, 64)

        total, pred, sigreg = loss_fn(predicted, target)

        assert total.ndim == 0
        assert pred.ndim == 0
        assert sigreg.ndim == 0


class TestLeJEPAModel:
    """Test complete LeJEPA model with 3-part architecture."""

    @pytest.fixture
    def model(self) -> LeJEPA:
        """Create test model."""
        return LeJEPA(
            input_dim=29,
            premarket_dim=6,
            premarket_len=15,
            first15_len=15,
            d_model=64,
            nhead=4,
            num_layers=2,
            embedding_dim=32,
            max_context_len=75,
            dropout=0.1,
            lambda_reg=0.5,
            reg_type="vicreg",
        )

    @pytest.fixture
    def sample_data(self) -> dict:
        """Create sample data for testing."""
        batch_size = 32
        return {
            "x_premarket": torch.randn(batch_size, 15, 6),
            "x_first15": torch.randn(batch_size, 15, 29),
            "x_context_full": torch.randn(batch_size, 45, 29),
            "x_context_small": torch.randn(batch_size, 10, 29),
            "x_context_empty": torch.randn(batch_size, 0, 29),
            "x_target": torch.randn(batch_size, 15, 29),
        }

    def test_initialization(self, model: LeJEPA) -> None:
        """Test model initialization."""
        assert model.input_dim == 29
        assert model.premarket_dim == 6
        assert model.premarket_len == 15
        assert model.first15_len == 15
        assert model.max_context_len == 75

        # Check key components exist
        assert hasattr(model, "transformer")
        assert hasattr(model, "premarket_transformer")
        assert hasattr(model, "first15_projector")
        assert hasattr(model, "fusion")
        assert hasattr(model, "predictor")
        assert hasattr(model, "no_context_embedding")

    def test_forward_without_premarket(
        self, model: LeJEPA, sample_data: dict
    ) -> None:
        """Test forward pass without premarket (legacy mode)."""
        output = model(
            x_context=sample_data["x_context_full"],
            x_target=sample_data["x_target"],
            return_loss=True,
        )

        assert "context_embedding" in output
        assert "predicted_embedding" in output
        assert "target_embedding" in output
        assert "loss" in output

        assert output["context_embedding"].shape == (32, 32)
        assert output["predicted_embedding"].shape == (32, 32)
        assert not torch.isnan(output["loss"])

    def test_forward_full_architecture(
        self, model: LeJEPA, sample_data: dict
    ) -> None:
        """Test forward pass with full 3-part architecture."""
        output = model(
            x_context=sample_data["x_context_full"],
            x_target=sample_data["x_target"],
            x_first15=sample_data["x_first15"],
            x_premarket=sample_data["x_premarket"],
            return_loss=True,
        )

        assert "premarket_embedding" in output
        assert "first15_embedding" in output
        assert output["premarket_embedding"].shape == (32, 32)
        assert output["first15_embedding"].shape == (32, 32)
        assert not torch.isnan(output["loss"])

    def test_forward_with_small_context(
        self, model: LeJEPA, sample_data: dict
    ) -> None:
        """Test forward pass with small context (10 min after first15)."""
        output = model(
            x_context=sample_data["x_context_small"],
            x_target=sample_data["x_target"],
            x_first15=sample_data["x_first15"],
            x_premarket=sample_data["x_premarket"],
            return_loss=True,
        )

        assert "context_embedding" in output
        assert not torch.isnan(output["loss"])

    def test_forward_empty_context(
        self, model: LeJEPA, sample_data: dict
    ) -> None:
        """Test forward pass with empty context (9:45 AM, first15 only)."""
        output = model(
            x_context=sample_data["x_context_empty"],
            x_target=sample_data["x_target"],
            x_first15=sample_data["x_first15"],
            x_premarket=sample_data["x_premarket"],
            return_loss=True,
        )

        # Should still produce valid embeddings using no_context_embedding
        assert output["context_embedding"].shape == (32, 32)
        assert not torch.isnan(output["loss"])

    def test_inference_mode(
        self, model: LeJEPA, sample_data: dict
    ) -> None:
        """Test inference mode (no target)."""
        model.eval()
        with torch.no_grad():
            output = model(
                x_context=sample_data["x_context_full"],
                x_first15=sample_data["x_first15"],
                x_premarket=sample_data["x_premarket"],
                return_loss=False,
            )

        assert "context_embedding" in output
        assert "predicted_embedding" in output
        assert "loss" not in output  # No loss in inference mode

    def test_gradient_flow(
        self, model: LeJEPA, sample_data: dict
    ) -> None:
        """Test gradients flow through all components."""
        output = model(
            x_context=sample_data["x_context_full"],
            x_target=sample_data["x_target"],
            x_first15=sample_data["x_first15"],
            x_premarket=sample_data["x_premarket"],
            return_loss=True,
        )

        output["loss"].backward()

        # Check gradients exist for key components
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.transformer.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.premarket_transformer.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.first15_projector.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.fusion.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.predictor.parameters())

    def test_training_step(
        self, model: LeJEPA, sample_data: dict
    ) -> None:
        """Test training_step convenience method."""
        total, pred, reg = model.training_step(
            x_context=sample_data["x_context_full"],
            x_target=sample_data["x_target"],
            x_first15=sample_data["x_first15"],
            x_premarket=sample_data["x_premarket"],
        )

        assert total.ndim == 0
        assert pred.ndim == 0
        assert reg.ndim == 0
        assert not torch.isnan(total)

    def test_encode_method(
        self, model: LeJEPA, sample_data: dict
    ) -> None:
        """Test encode convenience method."""
        embedding = model.encode(
            x_context=sample_data["x_context_full"],
            x_first15=sample_data["x_first15"],
            x_premarket=sample_data["x_premarket"],
        )

        assert embedding.shape == (32, 32)


class TestLeJEPACheckpoint:
    """Test checkpoint save/load functionality."""

    @pytest.fixture
    def model(self) -> LeJEPA:
        """Create test model."""
        return LeJEPA(
            input_dim=29,
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

    def test_save_load_checkpoint(self, model: LeJEPA) -> None:
        """Test checkpoint save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"

            # Save
            model.save_checkpoint(path, epoch=5, custom_value="test")

            # Load
            loaded_model, checkpoint = LeJEPA.load_checkpoint(path)

            assert checkpoint["epoch"] == 5
            assert checkpoint["custom_value"] == "test"

            # Check config was saved
            config = checkpoint["config"]
            assert config["input_dim"] == 29
            assert config["premarket_len"] == 15
            assert config["first15_len"] == 15
            assert config["max_context_len"] == 75

    def test_loaded_model_equivalence(self, model: LeJEPA) -> None:
        """Test loaded model produces same output as original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "checkpoint.pt"
            model.save_checkpoint(path, epoch=1)

            loaded_model, _ = LeJEPA.load_checkpoint(path)

            # Create test input
            x_context = torch.randn(8, 30, 29)
            x_first15 = torch.randn(8, 15, 29)
            x_premarket = torch.randn(8, 15, 6)

            # Compare outputs
            model.eval()
            loaded_model.eval()
            with torch.no_grad():
                output1 = model(
                    x_context=x_context,
                    x_first15=x_first15,
                    x_premarket=x_premarket,
                    return_loss=False,
                )
                output2 = loaded_model(
                    x_context=x_context,
                    x_first15=x_first15,
                    x_premarket=x_premarket,
                    return_loss=False,
                )

            diff = (output1["context_embedding"] - output2["context_embedding"]).abs().max()
            assert diff.item() < 1e-6


class TestLeJEPARegTypes:
    """Test both regularization types."""

    @pytest.mark.parametrize("reg_type", ["vicreg", "sigreg"])
    def test_reg_type(self, reg_type: str) -> None:
        """Test model works with both reg types."""
        model = LeJEPA(
            input_dim=29,
            premarket_dim=6,
            premarket_len=15,
            first15_len=15,
            d_model=64,
            nhead=4,
            num_layers=2,
            embedding_dim=32,
            max_context_len=75,
            reg_type=reg_type,
        )

        assert model.reg_type == reg_type

        # Forward pass
        output = model(
            x_context=torch.randn(32, 30, 29),
            x_target=torch.randn(32, 10, 29),
            x_first15=torch.randn(32, 15, 29),
            return_loss=True,
        )

        assert not torch.isnan(output["loss"])
        assert not torch.isnan(output["reg_loss"])
