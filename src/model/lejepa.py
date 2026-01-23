"""
LeJEPA: Latent-Euclidean Joint-Embedding Predictive Architecture for financial time series.

Architecture:
1. Input Projection: Market features → Transformer dimension
2. Positional Encoding: Learnable positional embeddings
3. Transformer Encoder: Self-attention over time steps
4. Projector: Mean-pooled output → State embedding
5. Predictor: Maps current state to future state
6. Loss: MSE(predicted, target) + λ * SIGReg(current_state)

Input Features (95 total):
- Core OHLCV + options + flow features (70 columns)
- Context columns (25): T-3, T-2, T-1, PM, OR OHLCV (5 each)
  - T-3/T-2/T-1/PM: Same value for all rows in a day
  - OR (Opening Range): NaN for first 14 rows, valid from row 14+
"""
from typing import Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.loss import SIGRegLoss, VICRegLoss


class LeJEPA(nn.Module):
    """
    Transformer-based LeJEPA for financial time series.

    Key design choices:
    - Transformer processes sequence of feature vectors (not flattened)
    - Learnable positional embeddings capture temporal patterns
    - Mean pooling over time steps for aggregation
    - Siamese encoder for context and target (shared weights)
    - SIGReg/VICReg forces state to be isotropic Gaussian (prevents collapse)

    Input Structure:
    - Each row has 95 features including:
      - Core features (70): OHLCV, options, flow, time features
      - Context columns (25): T-3, T-2, T-1, PM, OR OHLCV (5 each)

    Args:
        input_dim: Number of features per candle (95 with context columns)
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        embedding_dim: Latent state dimension (the "representation")
        max_context_len: Maximum context length
        dropout: Dropout probability
        lambda_reg: Weight for regularization loss (0.5 = equal weight pred/reg)
        reg_type: Regularization type - "vicreg" or "sigreg"
    """

    def __init__(
        self,
        input_dim: int = 95,  # 70 core + 25 context columns
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        embedding_dim: int = 64,
        max_context_len: int = 75,
        dropout: float = 0.1,
        lambda_reg: float = 0.5,
        reg_type: str = "vicreg",
    ) -> None:
        super().__init__()

        # Validate reg_type
        if reg_type not in ("sigreg", "vicreg"):
            raise ValueError(f"reg_type must be 'sigreg' or 'vicreg', got '{reg_type}'")

        # Store config for checkpoint save/load
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.max_context_len = max_context_len
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type

        # === Context Encoder ===
        # 1. Input Projection (Feature Vector -> Transformer Dimension)
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_context_len, d_model) * 0.02)

        # 3. Transformer Encoder (The Backbone)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # === Projector ===
        self.projector = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embedding_dim),
        )

        # Learnable "no context" embedding (used when context is empty)
        self.no_context_embedding = nn.Parameter(torch.randn(embedding_dim) * 0.02)

        # === Predictor ===
        # Maps current state to future state
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, embedding_dim),
        )

        # === Loss Components ===
        if reg_type == "sigreg":
            self.reg_loss = SIGRegLoss(embedding_dim=embedding_dim)
        else:
            self.reg_loss = VICRegLoss()

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values for stable training."""
        modules_to_init = [self.projector, self.predictor]
        for module in modules_to_init:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight, gain=0.1)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

    def forward_encoder(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode context sequence to embedding.

        Args:
            x: Context tensor [B, T, input_dim] where T is 0-max_context_len.

        Returns:
            Context embedding [B, embedding_dim]
        """
        # Handle empty context
        if x is None or x.shape[1] == 0:
            B = x.shape[0] if x is not None else 1
            return self.no_context_embedding.unsqueeze(0).expand(B, -1)

        B, T, F = x.shape

        # Project to transformer dimension
        h = self.input_proj(x)  # [B, T, d_model]

        # Add positional embedding
        h = h + self.pos_embedding[:, :T, :]

        # Run transformer
        h = self.transformer(h)  # [B, T, d_model]

        # Mean pool and project
        h = h.mean(dim=1)  # [B, d_model]
        return self.projector(h)  # [B, embedding_dim]

    def encode(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode context to embedding (alias for forward_encoder).

        Args:
            x: Context tensor [B, T, input_dim]

        Returns:
            Embedding [B, embedding_dim]
        """
        return self.forward_encoder(x)

    def forward(
        self,
        x_context: Optional[torch.Tensor] = None,
        x_target: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LeJEPA.

        Args:
            x_context: Context window [B, T, input_dim]
            x_target: Target (future) window [B, T_tgt, input_dim] (optional for inference)
            return_loss: Whether to compute and return loss

        Returns:
            Dictionary containing:
            - context_embedding: Current state [B, embedding_dim]
            - predicted_embedding: Predicted future state [B, embedding_dim]
            - target_embedding: Actual future state [B, embedding_dim] (if x_target)
            - loss: Total loss (if return_loss=True and x_target provided)
            - pred_loss: Prediction loss component
            - reg_loss: Regularization loss component
        """
        # 1. Encode current state
        state_current = self.forward_encoder(x_context)

        # 2. Predict future state
        state_predicted = self.predictor(state_current)

        output = {
            "context_embedding": state_current,
            "predicted_embedding": state_predicted,
        }

        # 3. If target provided, encode it and compute loss
        if x_target is not None:
            # Encode future state (no gradient - frozen for target)
            with torch.no_grad():
                state_target = self.forward_encoder(x_target)

            output["target_embedding"] = state_target

            if return_loss:
                # Prediction loss (MSE in latent space)
                pred_loss = F.mse_loss(state_predicted, state_target)

                # Regularization loss on current state representation
                reg_loss = self.reg_loss(state_current)

                # Total loss: pred_loss + lambda_reg * reg_loss
                total_loss = pred_loss + self.lambda_reg * reg_loss

                output["loss"] = total_loss
                output["pred_loss"] = pred_loss
                output["reg_loss"] = reg_loss
                output["sigreg_loss"] = reg_loss  # Backwards compat

        return output

    def training_step(
        self,
        x_context: Optional[torch.Tensor] = None,
        x_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Convenience method for training."""
        return self.forward(x_context, x_target, return_loss=True)

    def get_config(self) -> dict:
        """Get model configuration for saving."""
        return {
            "input_dim": self.input_dim,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "embedding_dim": self.embedding_dim,
            "max_context_len": self.max_context_len,
            "dropout": self.dropout,
            "lambda_reg": self.lambda_reg,
            "reg_type": self.reg_type,
        }

    @classmethod
    def from_config(cls, config: dict) -> "LeJEPA":
        """Create model from configuration."""
        return cls(**config)

    def save(self, path: str | Path) -> None:
        """Save model weights and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": self.get_config(),
            "state_dict": self.state_dict(),
        }, path)

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        global_step: int = 0,
        train_metrics: Optional[dict] = None,
        val_metrics: Optional[dict] = None,
    ) -> None:
        """Save full training checkpoint with optimizer and metrics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "config": self.get_config(),
            "state_dict": self.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        if train_metrics is not None:
            checkpoint["train_metrics"] = train_metrics
        if val_metrics is not None:
            checkpoint["val_metrics"] = val_metrics
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> "LeJEPA":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        model = cls.from_config(checkpoint["config"])

        # Handle state dicts from torch.compile() which add _orig_mod prefix
        state_dict = checkpoint["state_dict"]
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # Strip _orig_mod. prefix if present (from compiled models)
            new_key = key.replace("._orig_mod", "")
            cleaned_state_dict[new_key] = value

        model.load_state_dict(cleaned_state_dict)
        return model
