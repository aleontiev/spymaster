"""
Masked Gated Policy for Multi-Scale JEPA Fusion.

This module implements a policy network that fuses embeddings from multiple
JEPA models (short/medium/long timescales) with masking support for morning
trading when not all scales are available.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class MaskedGatedPolicy(nn.Module):
    """
    Policy network that fuses multiple JEPA embeddings with masking support.

    The model handles missing inputs (e.g., morning trading when longer timescales
    aren't available yet) by using a masking mechanism that forces attention weights
    of missing heads to zero.

    Architecture:
        1. Time context injection: Concatenate sin/cos time to each embedding
        2. Gating network: Compute importance scores for each scale
        3. Masked softmax: Force missing scales to zero attention
        4. Fusion: Weighted sum of valid embeddings
        5. Policy head: Output target position logits

    Inputs:
        emb_short: [B, emb_dim] - Short-term embedding (always available)
        emb_med: [B, emb_dim] - Medium-term embedding (0-padded if missing)
        emb_long: [B, emb_dim] - Long-term embedding (0-padded if missing)
        time_context: [B, 2] - Sin/cos time encoding
        mask: [B, 3] - 1 for active, 0 for missing

    Outputs:
        action_logits: [B, 3] - Target positions (Neutral/Long/Short)
        attn_weights: [B, 3] - Gating weights for exit logic attribution
    """

    # Target position classes
    NEUTRAL = 0  # Want zero exposure
    LONG = 1     # Want to be long
    SHORT = 2    # Want to be short

    def __init__(
        self,
        emb_dim: int = 64,
        time_dim: int = 2,
        hidden_dim: int = 256,
        n_positions: int = 3,
        n_scales: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize the MaskedGatedPolicy.

        Args:
            emb_dim: Dimension of each JEPA embedding (default: 64)
            time_dim: Dimension of time context (default: 2 for sin/cos)
            hidden_dim: Hidden dimension for MLP layers (default: 256)
            n_positions: Number of target positions (default: 3)
            n_scales: Number of JEPA scales (default: 3)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()

        self.emb_dim = emb_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.n_positions = n_positions
        self.n_scales = n_scales

        # Input dimension after time context concatenation
        self.input_dim = emb_dim + time_dim

        # Gating Network (The "Manager")
        # Takes concatenated time-contextualized embeddings
        self.gate_net = nn.Sequential(
            nn.Linear(self.input_dim * n_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_scales),  # Raw scores for each scale
        )

        # Policy Head (The "Executor")
        # Takes fused embedding to produce target position
        self.policy_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_positions),  # Target positions
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        emb_short: torch.Tensor,
        emb_med: torch.Tensor,
        emb_long: torch.Tensor,
        time_context: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with masked gating.

        Args:
            emb_short: [B, emb_dim] - Short-term embedding
            emb_med: [B, emb_dim] - Medium-term embedding (0 if missing)
            emb_long: [B, emb_dim] - Long-term embedding (0 if missing)
            time_context: [B, 2] - Sin/cos time encoding
            mask: [B, 3] - 1 for active, 0 for missing

        Returns:
            action_logits: [B, n_positions] - Target position logits
            attn_weights: [B, n_scales] - Gating attention weights
        """
        # A. Contextualize Embeddings
        # Append time to each embedding so the gate sees time-contextualized features
        e_s = torch.cat([emb_short, time_context], dim=1)  # [B, emb_dim + 2]
        e_m = torch.cat([emb_med, time_context], dim=1)    # [B, emb_dim + 2]
        e_l = torch.cat([emb_long, time_context], dim=1)   # [B, emb_dim + 2]

        # Stack for later fusion: [B, 3, input_dim]
        stacked = torch.stack([e_s, e_m, e_l], dim=1)

        # Flatten for gate input: [B, 3 * input_dim]
        flat_input = torch.cat([e_s, e_m, e_l], dim=1)

        # B. Calculate Raw Attention Scores
        raw_scores = self.gate_net(flat_input)  # [B, 3]

        # C. Apply Masking (Critical Step)
        # Set scores of missing heads to -infinity so Softmax makes them 0
        raw_scores = raw_scores.masked_fill(mask == 0, float("-inf"))

        # D. Softmax to get Attention Weights
        attn_weights = F.softmax(raw_scores, dim=1)  # [B, 3]

        # Handle edge case where all masks are 0 (shouldn't happen, but be safe)
        # This can result in NaN from softmax(-inf, -inf, -inf)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # E. Fuse Embeddings
        # Weighted sum: [B, 3, input_dim] * [B, 3, 1] -> sum over dim 1 -> [B, input_dim]
        fused_context = torch.sum(stacked * attn_weights.unsqueeze(-1), dim=1)

        # F. Get Action Logits
        action_logits = self.policy_net(fused_context)  # [B, n_positions]

        return action_logits, attn_weights

    def get_action(
        self,
        emb_short: torch.Tensor,
        emb_med: torch.Tensor,
        emb_long: torch.Tensor,
        time_context: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action with probabilities (convenience method for inference).

        Returns:
            action: [B] - Argmax action (0=Neutral, 1=Long, 2=Short)
            action_probs: [B, n_positions] - Softmax probabilities
            attn_weights: [B, n_scales] - Gating attention weights
        """
        logits, attn_weights = self.forward(
            emb_short, emb_med, emb_long, time_context, mask
        )
        action_probs = F.softmax(logits, dim=-1)
        action = torch.argmax(action_probs, dim=-1)
        return action, action_probs, attn_weights

    def get_config(self) -> Dict:
        """Get model configuration for serialization."""
        return {
            "emb_dim": self.emb_dim,
            "time_dim": self.time_dim,
            "hidden_dim": self.hidden_dim,
            "n_positions": self.n_positions,
            "n_scales": self.n_scales,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "MaskedGatedPolicy":
        """Create model from configuration."""
        return cls(**config)

    def save(self, path: str) -> None:
        """Save model to disk."""
        torch.save(
            {
                "config": self.get_config(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MaskedGatedPolicy":
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls.from_config(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


def get_time_context(sin_time: torch.Tensor, cos_time: torch.Tensor) -> torch.Tensor:
    """
    Create time context tensor from sin/cos time features.

    Args:
        sin_time: [B] or [B, 1] - Sine of time
        cos_time: [B] or [B, 1] - Cosine of time

    Returns:
        time_context: [B, 2] - Concatenated time features
    """
    if sin_time.dim() == 1:
        sin_time = sin_time.unsqueeze(-1)
    if cos_time.dim() == 1:
        cos_time = cos_time.unsqueeze(-1)
    return torch.cat([sin_time, cos_time], dim=-1)


class DirectPolicy(nn.Module):
    """
    Simple MLP policy that concatenates all embeddings directly.

    No attention/gating - just direct concatenation + MLP.
    This serves as a simpler baseline to compare against MaskedGatedPolicy.
    """

    def __init__(
        self,
        emb_dim: int = 64,
        time_dim: int = 2,  # Sin/cos time encoding (same as MaskedGatedPolicy)
        hidden_dim: int = 256,
        n_positions: int = 3,
        n_scales: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.n_positions = n_positions
        self.n_scales = n_scales

        # Total input: 3 * emb_dim + time_dim
        input_dim = n_scales * emb_dim + time_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_positions),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        emb_short: torch.Tensor,
        emb_med: torch.Tensor,
        emb_long: torch.Tensor,
        time_context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,  # Ignored for compatibility
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass - simple concatenation.

        Args:
            emb_short: [B, emb_dim] - Short-term embedding
            emb_med: [B, emb_dim] - Medium-term embedding
            emb_long: [B, emb_dim] - Long-term embedding
            time_context: [B, 2] - Sin/cos time encoding
            mask: [B, 3] - Ignored (for API compatibility)

        Returns:
            action_logits: [B, n_positions]
            attn_weights: [B, 3] - Equal weights (1/3 each) for compatibility
        """
        # Concatenate everything
        x = torch.cat([emb_short, emb_med, emb_long, time_context], dim=1)

        # Forward through MLP
        logits = self.net(x)

        # Return uniform attention for compatibility
        batch_size = emb_short.shape[0]
        attn_weights = torch.ones(batch_size, 3, device=emb_short.device) / 3

        return logits, attn_weights

    def get_action(
        self,
        emb_short: torch.Tensor,
        emb_med: torch.Tensor,
        emb_long: torch.Tensor,
        time_context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action with probabilities."""
        logits, attn_weights = self.forward(
            emb_short, emb_med, emb_long, time_context, mask
        )
        action_probs = F.softmax(logits, dim=-1)
        action = torch.argmax(action_probs, dim=-1)
        return action, action_probs, attn_weights

    def get_config(self) -> Dict:
        return {
            "emb_dim": self.emb_dim,
            "time_dim": self.time_dim,
            "hidden_dim": self.hidden_dim,
            "n_positions": self.n_positions,
            "n_scales": self.n_scales,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "DirectPolicy":
        return cls(**config)

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.get_config(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "DirectPolicy":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls.from_config(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


class MultiScalePolicy(nn.Module):
    """
    Multi-scale policy with three separate classifier heads.

    Each JEPA embedding (short/medium/long) gets its own classifier head
    trained on scale-appropriate labels. A gating network learns to weight
    the outputs based on which scale is most relevant.

    Architecture:
        1. Three classifier heads (one per scale)
        2. Gating network that weights the scale outputs
        3. Final prediction = weighted combination of scale predictions

    This allows each scale to specialize:
        - Short head: Sharp impulse moves (5-min horizon)
        - Medium head: Momentum continuation (15-min horizon)
        - Long head: Trending grinds (30-min horizon)
    """

    NEUTRAL = 0
    LONG = 1
    SHORT = 2

    def __init__(
        self,
        emb_dim: int = 64,
        time_dim: int = 2,
        hidden_dim: int = 128,
        n_positions: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.n_positions = n_positions

        # Input dim for each head: embedding + time context
        head_input_dim = emb_dim + time_dim

        # Three separate classifier heads
        self.short_head = self._make_head(head_input_dim, hidden_dim, n_positions, dropout)
        self.med_head = self._make_head(head_input_dim, hidden_dim, n_positions, dropout)
        self.long_head = self._make_head(head_input_dim, hidden_dim, n_positions, dropout)

        # Gating network: takes all embeddings + time, outputs weights for each scale
        gate_input_dim = 3 * emb_dim + time_dim
        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # Weights for short/med/long
        )

        self._init_weights()

    def _make_head(
        self, input_dim: int, hidden_dim: int, n_positions: int, dropout: float
    ) -> nn.Module:
        """Create a classifier head."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_positions),
        )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        emb_short: torch.Tensor,
        emb_med: torch.Tensor,
        emb_long: torch.Tensor,
        time_context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_head_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with separate heads and gating.

        Args:
            emb_short: [B, emb_dim] - Short-term embedding
            emb_med: [B, emb_dim] - Medium-term embedding
            emb_long: [B, emb_dim] - Long-term embedding
            time_context: [B, 2] - Sin/cos time encoding
            mask: [B, 3] - Availability mask (1=available, 0=missing)
            return_head_logits: Whether to return individual head logits

        Returns:
            combined_logits: [B, n_positions] - Weighted combination
            gate_weights: [B, 3] - Gating weights (softmax)
            head_logits: Dict with 'short', 'med', 'long' logits (if return_head_logits)
        """
        # Concatenate embeddings with time for each head
        short_input = torch.cat([emb_short, time_context], dim=1)
        med_input = torch.cat([emb_med, time_context], dim=1)
        long_input = torch.cat([emb_long, time_context], dim=1)

        # Get logits from each head
        short_logits = self.short_head(short_input)  # [B, n_positions]
        med_logits = self.med_head(med_input)        # [B, n_positions]
        long_logits = self.long_head(long_input)     # [B, n_positions]

        # Compute gating weights
        gate_input = torch.cat([emb_short, emb_med, emb_long, time_context], dim=1)
        gate_raw = self.gate_net(gate_input)  # [B, 3]

        # Apply mask if provided
        if mask is not None:
            gate_raw = gate_raw.masked_fill(mask == 0, float("-inf"))

        gate_weights = F.softmax(gate_raw, dim=1)  # [B, 3]
        gate_weights = torch.nan_to_num(gate_weights, nan=0.0)

        # Stack logits: [B, 3, n_positions]
        stacked_logits = torch.stack([short_logits, med_logits, long_logits], dim=1)

        # Weighted combination: [B, 3, 1] * [B, 3, n_positions] -> sum -> [B, n_positions]
        combined_logits = torch.sum(
            gate_weights.unsqueeze(-1) * stacked_logits, dim=1
        )

        if return_head_logits:
            head_logits = {
                "short": short_logits,
                "med": med_logits,
                "long": long_logits,
            }
            return combined_logits, gate_weights, head_logits

        return combined_logits, gate_weights, None

    def compute_loss(
        self,
        emb_short: torch.Tensor,
        emb_med: torch.Tensor,
        emb_long: torch.Tensor,
        time_context: torch.Tensor,
        labels_short: torch.Tensor,
        labels_med: torch.Tensor,
        labels_long: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        criterion: Optional[nn.Module] = None,
        class_weights: Optional[torch.Tensor] = None,
        head_weight: float = 1.0,
        combined_weight: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-scale loss.

        Loss = head_weight * (L_short + L_med + L_long) + combined_weight * L_combined

        Args:
            labels_short/med/long: Scale-specific target labels
            criterion: Loss function (default: CrossEntropyLoss)
            class_weights: Optional class weights for CrossEntropyLoss [N, L, S]
            head_weight: Weight for individual head losses
            combined_weight: Weight for combined prediction loss

        Returns:
            total_loss: Weighted sum of losses
            loss_dict: Individual loss components
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Forward pass with head logits
        combined_logits, gate_weights, head_logits = self.forward(
            emb_short, emb_med, emb_long, time_context, mask, return_head_logits=True
        )

        # Individual head losses
        loss_short = criterion(head_logits["short"], labels_short)
        loss_med = criterion(head_logits["med"], labels_med)
        loss_long = criterion(head_logits["long"], labels_long)

        # Combined loss using medium labels as primary target
        # (can be changed to a "consensus" label later)
        loss_combined = criterion(combined_logits, labels_med)

        # Total loss
        total_loss = head_weight * (loss_short + loss_med + loss_long) + combined_weight * loss_combined

        loss_dict = {
            "loss_short": loss_short,
            "loss_med": loss_med,
            "loss_long": loss_long,
            "loss_combined": loss_combined,
            "total_loss": total_loss,
        }

        return total_loss, loss_dict

    def get_action(
        self,
        emb_short: torch.Tensor,
        emb_med: torch.Tensor,
        emb_long: torch.Tensor,
        time_context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action with probabilities."""
        logits, gate_weights, _ = self.forward(
            emb_short, emb_med, emb_long, time_context, mask
        )
        action_probs = F.softmax(logits, dim=-1)
        action = torch.argmax(action_probs, dim=-1)
        return action, action_probs, gate_weights

    def get_config(self) -> Dict:
        return {
            "emb_dim": self.emb_dim,
            "time_dim": self.time_dim,
            "hidden_dim": self.hidden_dim,
            "n_positions": self.n_positions,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "MultiScalePolicy":
        return cls(**config)

    def save(self, path: str) -> None:
        torch.save(
            {
                "config": self.get_config(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MultiScalePolicy":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = cls.from_config(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        return model


def create_mask_from_time(
    time_to_close: torch.Tensor,
    short_available_after: float = 0.0,    # Always available
    med_available_after: float = 0.115,    # ~45 min into session (45/390)
    long_available_after: float = 0.231,   # ~90 min into session (90/390)
) -> torch.Tensor:
    """
    Create mask based on time of day.

    The mask indicates which JEPA scales have enough data to be valid.
    Early in the trading session, only short-term is available.

    Args:
        time_to_close: [B] - Normalized time to close (1.0 at open, 0.0 at close)
        short_available_after: Time threshold for short scale (default: always)
        med_available_after: Time threshold for medium scale
        long_available_after: Time threshold for long scale

    Returns:
        mask: [B, 3] - Binary mask (1=available, 0=missing)
    """
    # time_to_close=1.0 means market just opened, 0.0 means about to close
    # time_elapsed = 1.0 - time_to_close
    time_elapsed = 1.0 - time_to_close

    batch_size = time_to_close.shape[0]
    device = time_to_close.device

    mask = torch.zeros(batch_size, 3, device=device)
    mask[:, 0] = (time_elapsed >= short_available_after).float()  # Short always
    mask[:, 1] = (time_elapsed >= med_available_after).float()    # Med after ~45min
    mask[:, 2] = (time_elapsed >= long_available_after).float()   # Long after ~90min

    return mask
