"""
LeJEPA (Latent-Euclidean Joint-Embedding Predictive Architecture) for Financial Data.

Transformer-based implementation following the SIGReg paper approach:
- Transformer encoder processes time series as sequence of feature vectors
- Learnable positional embeddings for temporal awareness
- Mean pooling aggregation for state vector
- SIGReg regularization using moment-matching

Architecture:
1. Input Projection: Feature vector → Transformer dimension
2. Positional Embedding: Learnable temporal encoding
3. Transformer Encoder: Self-attention over time steps
4. Projector: Mean-pooled output → State embedding
5. Predictor: Maps current state to future state
6. Loss: MSE(predicted, target) + λ * SIGReg(current_state)
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
    - SIGReg forces state to be isotropic Gaussian (prevents collapse)

    Input Structure:

    1. Premarket prefix (15 candles, 6 features: OHLCV + time_to_close):
       - T-4, T-3, T-2, T-1 day range candles (4 candles)
       - Blank separators between sections
       - First 3 premarket candles (4:00-4:02 AM)
       - Premarket range candle
       - Last 3 premarket candles (9:27-9:29 AM)
       Total: 15 candles at timestamps 09:15-09:29 AM

    2. First 15 minutes (15 candles, 29 features) - ALWAYS INCLUDED:
       - 9:30-9:44 AM candles - the pivotal opening period with highest volume
       - Processed separately as an "anchor" for every prediction
       - Uses same 29-feature format as main context

    3. Main context (0-75 candles, 29 features) - VARIABLE LENGTH:
       - Contiguous candles from 9:45 AM onward
       - Grows from 0 to 75 as the day progresses
       - Always the most recent candles after first 15 (no gaps)

    Timeline examples (predictions start at 9:45 AM):
    - At 9:45 AM: premarket (15) + first15 (15) + context (0 candles)
    - At 10:00 AM: premarket (15) + first15 (15) + context (15 candles, 9:45-10:00)
    - At 10:30 AM: premarket (15) + first15 (15) + context (45 candles, 9:45-10:30)
    - At 11:00 AM: premarket (15) + first15 (15) + context (75 candles, 9:45-11:00, max)

    Args:
        input_dim: Number of input features per timestep:
            - Stock OHLCV (5): open, high, low, close, volume
            - VWAP (1): vwap
            - Options (8): atm_spread, net_premium_flow, call/put_strikes_active,
                           atm/otm_call_volume, atm/otm_put_volume
            - GEX Flow (22): net_gex, net_dex, gamma_flow, delta_flow, market_velocity,
                             zero_gex_price, zero_dex_price, positive_gws, negative_gws,
                             delta_support, delta_resistance, gamma_call_wall, gamma_put_wall,
                             dex_call_wall, dex_put_wall, anchored_vwap_z, vwap_divergence,
                             implied_volatility, call/put_buy/sell_volume (4)
            - Time (4): time_to_close, sin_time, cos_time, day_of_week
        premarket_dim: Number of features in premarket candles (6: OHLCV + time_to_close)
        premarket_len: Number of premarket candles (15)
        first15_len: Length of "first 15 minutes" anchor (15 candles, 9:30-9:44 AM)
        d_model: Transformer hidden dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        embedding_dim: Latent state dimension (the "representation")
        max_context_len: Maximum main context length (75 candles after first 15)
        dropout: Dropout probability
        lambda_reg: Weight for regularization loss (0.5 = equal weight pred/reg)
        reg_type: Regularization type - "vicreg" or "sigreg"
    """

    def __init__(
        self,
        input_dim: int = 29,  # 5 OHLCV + 1 VWAP + 8 options + 11 GEX + 4 time
        premarket_dim: int = 6,  # OHLCV + time_to_close
        premarket_len: int = 15,  # 15 premarket candles
        first15_len: int = 15,  # First 15 minutes anchor (9:30-9:44)
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        embedding_dim: int = 64,
        max_context_len: int = 75,  # Max 75 candles after first 15 (9:45-11:00)
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
        self.first15_len = first15_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.max_context_len = max_context_len
        self.dropout = dropout
        self.lambda_reg = lambda_reg
        self.reg_type = reg_type

        # === Market Context Encoder (shared for first15 and main context) ===
        # 1. Input Projection (Feature Vector -> Transformer Dimension)
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding for main context (0-75 candles after first 15)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_context_len, d_model) * 0.02)

        # 3. Positional Encoding for first 15 minutes (separate positions)
        self.first15_pos_embedding = nn.Parameter(torch.randn(1, first15_len, d_model) * 0.02)

        # 4. The Transformer Encoder (The Backbone) - shared for first15 and context
        # Note: No masking needed - first15 is always complete, main context is variable length
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

        # === Premarket Prefix Encoder (15 candles, 6 features) ===
        # Stationary for the day: T-4/T-3/T-2/T-1 ranges + premarket candles
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

        # 5. Projectors (Transformer Output -> State Embedding)
        # First15 projector (separate from main context)
        self.first15_projector = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embedding_dim),
        )
        # Main context projector
        self.projector = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embedding_dim),
        )

        # 6. Learnable "no context" embedding (used when main context is empty)
        self.no_context_embedding = nn.Parameter(torch.randn(embedding_dim) * 0.02)

        # 7. Fusion layer - combines premarket + first15 + context embeddings
        # Input: embedding_dim * 3 (premarket + first15 + context) -> embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, embedding_dim),
        )

        # 8. Predictor (Maps fused state to future state)
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
        # Projector and fusion use smaller init for residual-like behavior
        modules_to_init = [
            self.projector, self.first15_projector, self.premarket_projector,
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
        Encode the 15-candle premarket prefix.

        The premarket prefix provides daily context:
        - T-4, T-3, T-2, T-1 day range candles
        - Premarket session data (first 3, range, last 3 candles)
        - Timestamps: 09:15-09:29 AM (synthetic for storage)

        Args:
            x_premarket: Premarket tensor [B, 15, 6] (OHLCV + time_to_close)

        Returns:
            Premarket embedding [B, embedding_dim]
        """
        B = x_premarket.shape[0]

        # Project to transformer dimension
        x = self.premarket_proj(x_premarket)  # [B, 15, d_model]

        # Add positional embedding
        x = x + self.premarket_pos_embedding

        # Run through premarket transformer
        x = self.premarket_transformer(x)  # [B, 15, d_model]

        # Mean pool and project
        premarket_vector = x.mean(dim=1)  # [B, d_model]
        return self.premarket_projector(premarket_vector)  # [B, embedding_dim]

    def forward_first15_encoder(self, x_first15: torch.Tensor) -> torch.Tensor:
        """
        Encode the first 15 minutes of trading (9:30-9:44 AM).

        This is the pivotal opening period with highest volume. It's processed
        separately and ALWAYS included in every prediction as an anchor.
        Uses the same transformer as main context but with separate positional
        embeddings and projector.

        Args:
            x_first15: First 15 candles [B, 15, 29] (same features as context)

        Returns:
            First15 embedding [B, embedding_dim]
        """
        B = x_first15.shape[0]

        # Project to transformer dimension (shared projection with context)
        x = self.input_proj(x_first15)  # [B, 15, d_model]

        # Add first15-specific positional embedding
        x = x + self.first15_pos_embedding

        # Run through shared transformer
        x = self.transformer(x)  # [B, 15, d_model]

        # Mean pool and project with first15-specific projector
        first15_vector = x.mean(dim=1)  # [B, d_model]
        return self.first15_projector(first15_vector)  # [B, embedding_dim]

    def forward_context_encoder(
        self,
        x_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode the main market context (0-75 candles AFTER first 15).

        The main context covers 9:45 AM - 11:00 AM. This is the variable-length
        portion that grows as the day progresses. Always contiguous - starts
        at 9:45 and extends to current time.

        Args:
            x_context: Main context tensor [B, T, 29] where T is 0-75.
                       Can be None or have T=0 at 9:45 AM.

        Returns:
            Context embedding [B, embedding_dim]
        """
        # Handle empty context (9:45 AM, only have first 15)
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
        first15_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode market context and fuse with premarket + first15 embeddings.

        This is the main encoding function that combines:
        1. Premarket embedding (from 15 basic candles, 6 features)
        2. First15 embedding (from first 15 market candles, 29 features) - REQUIRED
        3. Context embedding (from 0-75 enhanced candles after first 15)

        Args:
            x_context: Main context tensor [B, 0-75, 29]. Can be empty.
            premarket_embedding: Pre-computed premarket embedding [B, embedding_dim].
            first15_embedding: Pre-computed first15 embedding [B, embedding_dim].
                               Required for full mode.

        Returns:
            Fused state embedding [B, embedding_dim]
        """
        # Encode main market context
        context_embedding = self.forward_context_encoder(x_context)

        # If both premarket and first15 provided, do 3-way fusion
        if premarket_embedding is not None and first15_embedding is not None:
            fused = torch.cat([premarket_embedding, first15_embedding, context_embedding], dim=-1)
            return self.fusion(fused)  # [B, embedding_dim]

        # Legacy mode: only context (for backwards compatibility)
        return context_embedding

    def encode(
        self,
        x_context: Optional[torch.Tensor] = None,
        x_first15: Optional[torch.Tensor] = None,
        x_premarket: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full encoding: premarket + first15 + market context -> fused embedding.

        Args:
            x_context: Main context [B, 0-75, 29] (can be empty, after first 15)
            x_first15: First 15 minutes [B, 15, 29] (REQUIRED for full mode)
            x_premarket: Premarket candles [B, 15, 6]

        Returns:
            Final fused embedding [B, embedding_dim]
        """
        premarket_embedding = None
        first15_embedding = None

        if x_premarket is not None:
            premarket_embedding = self.forward_premarket_encoder(x_premarket)

        if x_first15 is not None:
            first15_embedding = self.forward_first15_encoder(x_first15)

        return self.forward_encoder(x_context, premarket_embedding, first15_embedding)

    def forward(
        self,
        x_context: Optional[torch.Tensor] = None,
        x_target: Optional[torch.Tensor] = None,
        x_first15: Optional[torch.Tensor] = None,
        x_premarket: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LeJEPA.

        Architecture:
        1. Premarket prefix (15 basic candles, 6 features)
        2. First 15 minutes (15 candles, 29 features) - ALWAYS included as anchor
        3. Main context (0-75 candles, 29 features) - contiguous, grows with time
        4. 3-way fusion of premarket + first15 + context embeddings
        5. Prediction head for future state

        Args:
            x_context: Main context [B, 0-75, 29] - contiguous candles from 9:45 AM.
                       Grows from 0 to 75 as day progresses. Empty at 9:45 AM.
            x_target: Target (future) window [B, T_tgt, 29] (optional for inference)
            x_first15: First 15 minutes [B, 15, 29] - 9:30-9:44 AM (REQUIRED).
                       The pivotal opening period, always included as anchor.
            x_premarket: Premarket candles [B, 15, 6] - T-4/T-3/T-2/T-1 + premarket
            return_loss: Whether to compute and return loss

        Returns:
            Dictionary containing:
            - context_embedding: Current state [B, embedding_dim]
            - predicted_embedding: Predicted future state [B, embedding_dim]
            - premarket_embedding: Premarket embedding [B, embedding_dim] (if provided)
            - first15_embedding: First15 embedding [B, embedding_dim] (if provided)
            - target_embedding: Actual future state [B, embedding_dim] (if x_target)
            - loss: Total loss (if return_loss=True and x_target provided)
            - pred_loss: Prediction loss component
            - reg_loss: Regularization loss component
            - sigreg_loss: Alias for reg_loss (backwards compat)
        """
        # 1. Encode premarket if provided
        premarket_embedding = None
        if x_premarket is not None:
            premarket_embedding = self.forward_premarket_encoder(x_premarket)

        # 2. Encode first 15 minutes if provided (the anchor)
        first15_embedding = None
        if x_first15 is not None:
            first15_embedding = self.forward_first15_encoder(x_first15)

        # 3. Encode current state (premarket + first15 + main context)
        state_current = self.forward_encoder(x_context, premarket_embedding, first15_embedding)

        # 4. Predict future state
        state_predicted = self.predictor(state_current)

        output = {
            "context_embedding": state_current,
            "predicted_embedding": state_predicted,
        }

        # Include component embeddings if computed
        if premarket_embedding is not None:
            output["premarket_embedding"] = premarket_embedding
        if first15_embedding is not None:
            output["first15_embedding"] = first15_embedding

        # 5. If target provided, encode it and compute loss
        if x_target is not None:
            # Encode future state (no gradient - frozen for target)
            # Target is encoded as main context with premarket + first15 for consistency
            with torch.no_grad():
                state_target = self.forward_encoder(x_target, premarket_embedding, first15_embedding)

            output["target_embedding"] = state_target

            if return_loss:
                # Prediction loss (MSE in latent space)
                pred_loss = F.mse_loss(state_predicted, state_target)

                # Regularization loss on current state representation
                reg_loss = self.reg_loss(state_current)

                # Total loss with lambda_reg weight
                total_loss = (1 - self.lambda_reg) * pred_loss + self.lambda_reg * reg_loss

                output["loss"] = total_loss
                output["pred_loss"] = pred_loss
                output["reg_loss"] = reg_loss
                output["sigreg_loss"] = reg_loss  # Backwards compat

        return output

    def training_step(
        self,
        x_context: Optional[torch.Tensor] = None,
        x_target: Optional[torch.Tensor] = None,
        x_first15: Optional[torch.Tensor] = None,
        x_premarket: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method for training loop.

        Args:
            x_context: Main context window [B, 0-75, 29] (after first 15)
            x_target: Target window [B, T_tgt, 29]
            x_first15: First 15 minutes [B, 15, 29] - the anchor
            x_premarket: Premarket candles [B, 15, 6]

        Returns:
            Tuple of (total_loss, pred_loss, reg_loss)
        """
        output = self.forward(
            x_context=x_context,
            x_target=x_target,
            x_first15=x_first15,
            x_premarket=x_premarket,
            return_loss=True,
        )
        return output["loss"], output["pred_loss"], output["sigreg_loss"]

    def save_checkpoint(self, path: Path, epoch: int, **kwargs) -> None:
        """
        Save model checkpoint with config.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
            **kwargs: Additional items to save (optimizer, scheduler, etc.)
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "premarket_dim": self.premarket_dim,
                "premarket_len": self.premarket_len,
                "first15_len": self.first15_len,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "embedding_dim": self.embedding_dim,
                "max_context_len": self.max_context_len,
                "dropout": self.dropout,
                "lambda_reg": self.lambda_reg,
                "reg_type": self.reg_type,
            },
            **kwargs,
        }
        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved to {path}")

    @classmethod
    def load_checkpoint(
        cls, path: Path, device: Optional[torch.device] = None
    ) -> Tuple["LeJEPA", Dict]:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load model to

        Returns:
            Tuple of (model, checkpoint_dict)
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]

        # Initialize model
        model = cls(
            input_dim=config.get("input_dim", 29),
            premarket_dim=config.get("premarket_dim", 6),
            premarket_len=config.get("premarket_len", 15),
            first15_len=config.get("first15_len", 15),
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 4),
            embedding_dim=config.get("embedding_dim", 64),
            max_context_len=config.get("max_context_len", 75),
            dropout=config.get("dropout", 0.1),
            lambda_reg=config.get("lambda_reg", 0.5),
            reg_type=config.get("reg_type", "vicreg"),
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])

        if device:
            model.to(device)

        print(f"✓ Model loaded from {path} (epoch {checkpoint['epoch']})")
        return model, checkpoint


def test_lejepa() -> None:
    """Test the Transformer-based LeJEPA model."""
    print("=" * 60)
    print("Testing LeJEPA: Premarket (15) + First15 (15) + Context (0-75)")
    print("=" * 60)

    batch_size = 64
    input_dim = 29  # Enhanced features
    premarket_dim = 6  # Basic features (OHLCV + time_to_close)
    premarket_len = 15
    first15_len = 15  # First 15 minutes anchor
    max_context_len = 75  # After first 15 (9:45-11:00 AM)
    embedding_dim = 64

    # Create sample data
    x_premarket = torch.randn(batch_size, premarket_len, premarket_dim)
    x_first15 = torch.randn(batch_size, first15_len, input_dim)  # Anchor
    x_context_full = torch.randn(batch_size, 45, input_dim)  # 45 min after first15
    x_context_small = torch.randn(batch_size, 10, input_dim)  # 10 min after first15
    x_context_empty = torch.randn(batch_size, 0, input_dim)  # At 9:45 AM
    x_target = torch.randn(batch_size, 15, input_dim)

    print(f"\nInput shapes:")
    print(f"  Premarket: {x_premarket.shape} (15 basic candles, 6 features)")
    print(f"  First15: {x_first15.shape} (15 anchor candles, 9:30-9:44)")
    print(f"  Context full: {x_context_full.shape} (45 candles after first15)")
    print(f"  Context small: {x_context_small.shape} (10 candles)")
    print(f"  Context empty: {x_context_empty.shape} (at 9:45)")
    print(f"  Target: {x_target.shape}")

    # Test both reg types
    for reg_type in ["vicreg", "sigreg"]:
        print(f"\n{'='*60}")
        print(f"Testing LeJEPA with {reg_type.upper()}")
        print("=" * 60)

        model = LeJEPA(
            input_dim=input_dim,
            premarket_dim=premarket_dim,
            premarket_len=premarket_len,
            first15_len=first15_len,
            d_model=128,
            nhead=4,
            num_layers=4,
            embedding_dim=embedding_dim,
            max_context_len=max_context_len,
            lambda_reg=0.5,
            reg_type=reg_type,
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Reg type: {model.reg_type}")
        print(f"Config: premarket={model.premarket_len}, first15={model.first15_len}, context={model.max_context_len}")

        # Test 1: Forward pass (legacy mode - only context)
        print("\n1. Forward pass (legacy mode - only context)...")
        output = model(x_context=x_context_full, x_target=x_target, return_loss=True)

        print(f"Output keys: {list(output.keys())}")
        print(f"Context embedding: {output['context_embedding'].shape}")
        print(f"\nLosses:")
        print(f"  Total: {output['loss'].item():.6f}")
        print(f"  Prediction: {output['pred_loss'].item():.6f}")
        print(f"  Reg ({reg_type}): {output['reg_loss'].item():.6f}")

        # Test 2: Forward pass with full 3-part architecture
        print("\n2. Forward pass (full: premarket + first15 + context)...")
        model.zero_grad()
        output_full = model(
            x_context=x_context_full,
            x_target=x_target,
            x_first15=x_first15,
            x_premarket=x_premarket,
            return_loss=True,
        )

        print(f"Output keys: {list(output_full.keys())}")
        print(f"Premarket embedding: {output_full['premarket_embedding'].shape}")
        print(f"First15 embedding: {output_full['first15_embedding'].shape}")
        print(f"\nLosses with full context:")
        print(f"  Total: {output_full['loss'].item():.6f}")

        # Test 3: Forward pass with small context (10 min after first15)
        print("\n3. Forward pass (small context - 10 min after first15)...")
        model.zero_grad()
        output_small = model(
            x_context=x_context_small,
            x_target=x_target,
            x_first15=x_first15,
            x_premarket=x_premarket,
            return_loss=True,
        )

        print(f"Small context embedding: {output_small['context_embedding'].shape}")
        print(f"Losses:")
        print(f"  Total: {output_small['loss'].item():.6f}")

        # Test 4: Forward pass with EMPTY context (at 9:45 AM)
        print("\n4. Forward pass (empty context - 9:45 AM, first15 only)...")
        model.zero_grad()
        output_empty = model(
            x_context=x_context_empty,
            x_target=x_target,
            x_first15=x_first15,
            x_premarket=x_premarket,
            return_loss=True,
        )

        print(f"Empty context embedding shape: {output_empty['context_embedding'].shape}")
        print(f"Losses with only first15:")
        print(f"  Total: {output_empty['loss'].item():.6f}")

        # Test gradient flow
        print("\n5. Testing gradient flow...")
        loss = output_full["loss"]
        loss.backward()

        transformer_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.transformer.parameters()
        )
        premarket_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.premarket_transformer.parameters()
        )
        first15_proj_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.first15_projector.parameters()
        )
        fusion_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.fusion.parameters()
        )
        predictor_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.predictor.parameters()
        )

        print(f"  Main transformer has gradients: {transformer_has_grad}")
        print(f"  Premarket transformer has gradients: {premarket_has_grad}")
        print(f"  First15 projector has gradients: {first15_proj_has_grad}")
        print(f"  Fusion layer has gradients: {fusion_has_grad}")
        print(f"  Predictor has gradients: {predictor_has_grad}")

        # Test training step
        print("\n6. Testing training_step()...")
        model.zero_grad()
        total, pred, reg = model.training_step(
            x_context=x_context_full,
            x_target=x_target,
            x_first15=x_first15,
            x_premarket=x_premarket,
        )
        print(f"  Total: {total.item():.6f}")
        print(f"  Pred: {pred.item():.6f}")
        print(f"  Reg: {reg.item():.6f}")

        # Test inference mode
        print("\n7. Testing inference mode...")
        model.eval()
        with torch.no_grad():
            output_inf = model(
                x_context=x_context_full,
                x_first15=x_first15,
                x_premarket=x_premarket,
                return_loss=False,
            )

        print(f"Inference output keys: {list(output_inf.keys())}")
        print(f"Predicted embedding: {output_inf['predicted_embedding'].shape}")

        # Test checkpoint save/load
        print("\n8. Testing checkpoint save/load...")
        checkpoint_path = Path(f"/tmp/test_lejepa_{reg_type}.pt")
        model.save_checkpoint(checkpoint_path, epoch=1, test_metadata="test_value")

        loaded_model, checkpoint = LeJEPA.load_checkpoint(checkpoint_path)
        print(f"  Loaded epoch: {checkpoint['epoch']}")
        print(f"  Config keys: {list(checkpoint['config'].keys())}")
        print(f"  Loaded reg_type: {loaded_model.reg_type}")
        print(f"  Loaded premarket_len: {loaded_model.premarket_len}")
        print(f"  Loaded first15_len: {loaded_model.first15_len}")
        print(f"  Loaded max_context_len: {loaded_model.max_context_len}")

        # Verify loaded model produces same output
        loaded_model.eval()
        with torch.no_grad():
            output_loaded = loaded_model(
                x_context=x_context_full,
                x_first15=x_first15,
                x_premarket=x_premarket,
                return_loss=False,
            )
            diff = (output_loaded['context_embedding'] - output_inf['context_embedding']).abs().max()
            print(f"  Max diff from original: {diff.item():.8f}")

        # Clean up
        checkpoint_path.unlink()

    print("\n" + "=" * 60)
    print("All LeJEPA tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_lejepa()
