"""
LeJEPA: Latent-Euclidean Joint-Embedding Predictive Architecture for financial time series.

Architecture:
1. Input Projection: Market features → Transformer dimension
2. Positional Encoding: Learnable positional embeddings
3. Transformer Encoder: Self-attention over time steps
4. Projector: Mean-pooled output → State embedding
5. Predictor: Maps current state to future state
6. Loss: MSE(predicted, target) + λ * SIGReg(current_state)

Input Structure:
- Premarket: 4 candles × 5 features (T-3, T-2, T-1, PM day ranges - OHLCV only)
- Opening Range: 1 candle × 5 features (aggregated OHLCV of 9:30-9:44 AM)
- Context: Variable candles × N features (full market data)
"""
from typing import Dict, Tuple, Optional
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
    1. Premarket prefix (4 candles, 5 features: OHLCV):
       - T-3: Day range candle (3 trading days ago)
       - T-2: Day range candle (2 trading days ago)
       - T-1: Day range candle (previous trading day)
       - PM: Premarket session range candle (4:00-9:29 AM)

    2. Opening Range (1 candle, 5 features: OHLCV):
       - Aggregated OHLCV of 9:30-9:44 AM (first 15 minutes)
       - High = max(high), Low = min(low), Volume = sum(volume)

    3. Main Context (variable length, N features):
       - Full market features from after opening range
       - Grows from 0 to max_context_len as day progresses

    Args:
        input_dim: Number of features per candle in main context
        premarket_dim: Number of features in premarket candles (5: OHLCV)
        premarket_len: Number of premarket candles (4: T-3, T-2, T-1, PM)
        opening_range_dim: Number of features in opening range (5: OHLCV)
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        embedding_dim: Latent state dimension (the "representation")
        max_context_len: Maximum main context length
        dropout: Dropout probability
        lambda_reg: Weight for regularization loss (0.5 = equal weight pred/reg)
        reg_type: Regularization type - "vicreg" or "sigreg"
    """

    def __init__(
        self,
        input_dim: int = 29,
        premarket_dim: int = 5,  # OHLCV only
        premarket_len: int = 4,  # T-3, T-2, T-1, PM
        opening_range_dim: int = 5,  # OHLCV only
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
        self.premarket_dim = premarket_dim
        self.premarket_len = premarket_len
        self.opening_range_dim = opening_range_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.max_context_len = max_context_len
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type

        # === Market Context Encoder ===
        # 1. Input Projection (Feature Vector -> Transformer Dimension)
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding for main context
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

        # === Premarket Encoder (4 candles, 5 features) ===
        self.premarket_proj = nn.Linear(premarket_dim, d_model)
        self.premarket_pos_embedding = nn.Parameter(
            torch.randn(1, premarket_len, d_model) * 0.02
        )
        # Lightweight premarket transformer (2 layers)
        premarket_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.premarket_transformer = nn.TransformerEncoder(
            premarket_encoder_layer, num_layers=2
        )
        self.premarket_projector = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embedding_dim),
        )

        # === Opening Range Encoder (1 candle, 5 features) ===
        # Simple MLP since it's just 1 candle
        self.opening_range_encoder = nn.Sequential(
            nn.Linear(opening_range_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, embedding_dim),
        )

        # === Context Projector ===
        self.projector = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embedding_dim),
        )

        # Learnable "no context" embedding (used when main context is empty)
        self.no_context_embedding = nn.Parameter(torch.randn(embedding_dim) * 0.02)

        # === Fusion Layer ===
        # Combines premarket + opening_range + context embeddings
        # Input: embedding_dim * 3 -> embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, embedding_dim),
        )

        # === Predictor ===
        # Maps fused state to future state
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
        modules_to_init = [
            self.projector, self.premarket_projector, self.opening_range_encoder,
            self.fusion, self.predictor
        ]
        for module in modules_to_init:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight, gain=0.1)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)

    def forward_premarket_encoder(self, x_premarket: torch.Tensor) -> torch.Tensor:
        """
        Encode the premarket candles (T-3, T-2, T-1, PM).

        Args:
            x_premarket: Premarket tensor [B, 4, 5] (OHLCV)

        Returns:
            Premarket embedding [B, embedding_dim]
        """
        B = x_premarket.shape[0]

        # Project to transformer dimension
        x = self.premarket_proj(x_premarket)  # [B, 4, d_model]

        # Add positional embedding
        x = x + self.premarket_pos_embedding

        # Run through transformer
        x = self.premarket_transformer(x)  # [B, 4, d_model]

        # Mean pool and project
        premarket_vector = x.mean(dim=1)  # [B, d_model]
        return self.premarket_projector(premarket_vector)  # [B, embedding_dim]

    def forward_opening_range_encoder(self, x_opening_range: torch.Tensor) -> torch.Tensor:
        """
        Encode the opening range candle (aggregated 9:30-9:44 AM).

        Args:
            x_opening_range: Opening range tensor [B, 5] (OHLCV)

        Returns:
            Opening range embedding [B, embedding_dim]
        """
        return self.opening_range_encoder(x_opening_range)

    def forward_context_encoder(
        self,
        x_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode the main market context.

        Args:
            x_context: Main context tensor [B, T, input_dim] where T is 0-max_context_len.

        Returns:
            Context embedding [B, embedding_dim]
        """
        # Handle empty context
        if x_context is None or x_context.shape[1] == 0:
            B = x_context.shape[0] if x_context is not None else 1
            return self.no_context_embedding.unsqueeze(0).expand(B, -1)

        B, T, F = x_context.shape

        # Project to transformer dimension
        x = self.input_proj(x_context)  # [B, T, d_model]

        # Add positional embedding
        x = x + self.pos_embedding[:, :T, :]

        # Run transformer
        x = self.transformer(x)  # [B, T, d_model]

        # Mean pool and project
        context_vector = x.mean(dim=1)  # [B, d_model]
        return self.projector(context_vector)  # [B, embedding_dim]

    def forward_encoder(
        self,
        x_context: Optional[torch.Tensor] = None,
        premarket_embedding: Optional[torch.Tensor] = None,
        opening_range_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode market context and fuse with premarket + opening_range embeddings.

        Args:
            x_context: Main context tensor [B, T, input_dim].
            premarket_embedding: Pre-computed premarket embedding [B, embedding_dim].
            opening_range_embedding: Pre-computed opening range embedding [B, embedding_dim].

        Returns:
            Fused state embedding [B, embedding_dim]
        """
        # Encode main market context
        context_embedding = self.forward_context_encoder(x_context)

        # If both premarket and opening_range provided, do 3-way fusion
        if premarket_embedding is not None and opening_range_embedding is not None:
            fused = torch.cat([premarket_embedding, opening_range_embedding, context_embedding], dim=-1)
            return self.fusion(fused)  # [B, embedding_dim]

        # Legacy mode: only context (for backwards compatibility)
        return context_embedding

    def encode(
        self,
        x_context: Optional[torch.Tensor] = None,
        x_opening_range: Optional[torch.Tensor] = None,
        x_premarket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full encoding: premarket + opening_range + context -> fused embedding.

        Args:
            x_context: Main context [B, T, input_dim]
            x_opening_range: Opening range [B, 5] (OHLCV)
            x_premarket: Premarket candles [B, 4, 5] (OHLCV)

        Returns:
            Final fused embedding [B, embedding_dim]
        """
        premarket_embedding = None
        opening_range_embedding = None

        if x_premarket is not None:
            premarket_embedding = self.forward_premarket_encoder(x_premarket)

        if x_opening_range is not None:
            opening_range_embedding = self.forward_opening_range_encoder(x_opening_range)

        return self.forward_encoder(x_context, premarket_embedding, opening_range_embedding)

    def forward(
        self,
        x_context: Optional[torch.Tensor] = None,
        x_target: Optional[torch.Tensor] = None,
        x_opening_range: Optional[torch.Tensor] = None,
        x_premarket: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LeJEPA.

        Args:
            x_context: Main context [B, T, input_dim]
            x_target: Target (future) window [B, T_tgt, input_dim] (optional for inference)
            x_opening_range: Opening range [B, 5] (OHLCV of 9:30-9:44 AM)
            x_premarket: Premarket candles [B, 4, 5] (T-3, T-2, T-1, PM)
            return_loss: Whether to compute and return loss

        Returns:
            Dictionary containing:
            - context_embedding: Current state [B, embedding_dim]
            - predicted_embedding: Predicted future state [B, embedding_dim]
            - premarket_embedding: Premarket embedding [B, embedding_dim] (if provided)
            - opening_range_embedding: Opening range embedding [B, embedding_dim] (if provided)
            - target_embedding: Actual future state [B, embedding_dim] (if x_target)
            - loss: Total loss (if return_loss=True and x_target provided)
            - pred_loss: Prediction loss component
            - reg_loss: Regularization loss component
        """
        # 1. Encode premarket if provided
        premarket_embedding = None
        if x_premarket is not None:
            premarket_embedding = self.forward_premarket_encoder(x_premarket)

        # 2. Encode opening range if provided
        opening_range_embedding = None
        if x_opening_range is not None:
            opening_range_embedding = self.forward_opening_range_encoder(x_opening_range)

        # 3. Encode current state (premarket + opening_range + context)
        state_current = self.forward_encoder(x_context, premarket_embedding, opening_range_embedding)

        # 4. Predict future state
        state_predicted = self.predictor(state_current)

        output = {
            "context_embedding": state_current,
            "predicted_embedding": state_predicted,
        }

        # Include component embeddings if computed
        if premarket_embedding is not None:
            output["premarket_embedding"] = premarket_embedding
        if opening_range_embedding is not None:
            output["opening_range_embedding"] = opening_range_embedding

        # 5. If target provided, encode it and compute loss
        if x_target is not None:
            # Encode future state (no gradient - frozen for target)
            with torch.no_grad():
                state_target = self.forward_encoder(x_target, premarket_embedding, opening_range_embedding)

            output["target_embedding"] = state_target

            if return_loss:
                # Prediction loss (MSE in latent space)
                pred_loss = F.mse_loss(state_predicted, state_target)

                # Regularization loss on current state representation
                reg_loss = self.reg_loss(state_current)

                # Total loss: pred_loss + lambda_reg * reg_loss
                # lambda_reg controls relative weight of regularization
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
        x_opening_range: Optional[torch.Tensor] = None,
        x_premarket: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Convenience method for training."""
        return self.forward(x_context, x_target, x_opening_range, x_premarket, return_loss=True)

    def get_config(self) -> dict:
        """Get model configuration for saving."""
        return {
            "input_dim": self.input_dim,
            "premarket_dim": self.premarket_dim,
            "premarket_len": self.premarket_len,
            "opening_range_dim": self.opening_range_dim,
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
        checkpoint = torch.load(path, map_location=map_location)
        model = cls.from_config(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model
