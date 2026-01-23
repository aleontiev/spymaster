"""
Masked Gated Policy Executor for Production Trading.

This module provides the execution wrapper for running the MaskedGatedPolicy
in production, handling:
- Model loading and inference
- Time-based mask computation
- Exit logic based on attention weights
"""

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from src.model.lejepa import LeJEPA
from src.model.masked_gated_policy import MaskedGatedPolicy, get_time_context


class TargetPosition(IntEnum):
    """Target position enum matching the policy output."""
    NEUTRAL = 0
    LONG = 1
    SHORT = 2


@dataclass
class TradeSignal:
    """Trade signal with exit parameters."""
    position: TargetPosition
    confidence: float
    attn_weights: Tuple[float, float, float]  # (short, med, long)
    dominant_scale: str  # "short", "medium", or "long"
    time_limit_mins: int  # Suggested time limit based on dominant scale


@dataclass
class ExitParameters:
    """Exit parameters for a trade based on attention weights."""
    stop_loss_atr_mult: float = 1.0
    take_profit_atr_mult: float = 2.0
    time_limit_mins: int = 30


class MaskedGatedExecutor:
    """
    Production executor for MaskedGatedPolicy.

    Handles loading models, computing masks based on time, and generating
    trade signals with appropriate exit parameters.

    Example usage:
        executor = MaskedGatedExecutor.from_checkpoints(
            policy_path="checkpoints/policy_best.pt",
            short_jepa_path="checkpoints/lejepa-15-5/lejepa_best.pt",
            med_jepa_path="checkpoints/lejepa-45-15/lejepa_best.pt",
            long_jepa_path="checkpoints/lejepa-90-30/lejepa_best.pt",
        )

        # Get signal for current market state
        signal = executor.get_signal(
            context_short=short_context_tensor,
            context_med=med_context_tensor,
            context_long=long_context_tensor,
            time_to_close=0.5,  # Normalized, 0.5 = mid-day
            sin_time=0.0,
            cos_time=1.0,
        )
    """

    # Scale names and their index
    SCALE_NAMES = ["short", "medium", "long"]

    # Time limits based on dominant attention (in minutes)
    TIME_LIMITS = {
        "short": 10,    # Scalp
        "medium": 45,   # Swing
        "long": 120,    # Trend
    }

    # Minimum time elapsed (as fraction of day) for each scale
    # Market day = 390 minutes
    SCALE_AVAILABILITY = {
        "short": 0.0,     # Always available
        "medium": 0.115,  # ~45 min / 390 min
        "long": 0.231,    # ~90 min / 390 min
    }

    def __init__(
        self,
        policy: MaskedGatedPolicy,
        short_jepa: LeJEPA,
        med_jepa: LeJEPA,
        long_jepa: LeJEPA,
        device: str = "cuda",
    ):
        """
        Initialize the executor.

        Args:
            policy: Trained MaskedGatedPolicy model
            short_jepa: Short-term JEPA encoder
            med_jepa: Medium-term JEPA encoder
            long_jepa: Long-term JEPA encoder
            device: Device for inference
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.policy = policy.to(self.device)
        self.policy.eval()

        self.short_jepa = short_jepa.to(self.device)
        self.short_jepa.eval()

        self.med_jepa = med_jepa.to(self.device)
        self.med_jepa.eval()

        self.long_jepa = long_jepa.to(self.device)
        self.long_jepa.eval()

        # Freeze all models
        for model in [self.policy, self.short_jepa, self.med_jepa, self.long_jepa]:
            for param in model.parameters():
                param.requires_grad = False

    @classmethod
    def from_checkpoints(
        cls,
        policy_path: str,
        short_jepa_path: str,
        med_jepa_path: str,
        long_jepa_path: str,
        device: str = "cuda",
    ) -> "MaskedGatedExecutor":
        """
        Create executor from checkpoint paths.

        Args:
            policy_path: Path to trained MaskedGatedPolicy
            short_jepa_path: Path to short-term JEPA
            med_jepa_path: Path to medium-term JEPA
            long_jepa_path: Path to long-term JEPA
            device: Device for inference

        Returns:
            Configured MaskedGatedExecutor
        """
        policy = MaskedGatedPolicy.load(policy_path, device=device)
        short_jepa = LeJEPA.load(short_jepa_path, map_location=device)
        med_jepa = LeJEPA.load(med_jepa_path, map_location=device)
        long_jepa = LeJEPA.load(long_jepa_path, map_location=device)

        return cls(policy, short_jepa, med_jepa, long_jepa, device)

    def compute_mask(self, time_to_close: float) -> torch.Tensor:
        """
        Compute availability mask based on time of day.

        Args:
            time_to_close: Normalized time to close (1.0 at open, 0.0 at close)

        Returns:
            mask: [1, 3] tensor with 1 for available scales
        """
        time_elapsed = 1.0 - time_to_close

        mask = torch.zeros(1, 3, device=self.device)
        mask[0, 0] = float(time_elapsed >= self.SCALE_AVAILABILITY["short"])
        mask[0, 1] = float(time_elapsed >= self.SCALE_AVAILABILITY["medium"])
        mask[0, 2] = float(time_elapsed >= self.SCALE_AVAILABILITY["long"])

        return mask

    def get_dominant_scale(self, attn_weights: torch.Tensor) -> Tuple[str, int]:
        """
        Determine the dominant scale from attention weights.

        Args:
            attn_weights: [1, 3] attention weights

        Returns:
            Tuple of (scale_name, time_limit_mins)
        """
        weights = attn_weights[0].cpu().tolist()
        max_idx = weights.index(max(weights))
        scale_name = self.SCALE_NAMES[max_idx]
        time_limit = self.TIME_LIMITS[scale_name]
        return scale_name, time_limit

    @torch.no_grad()
    def get_signal(
        self,
        context_short: torch.Tensor,
        context_med: Optional[torch.Tensor],
        context_long: Optional[torch.Tensor],
        time_to_close: float,
        sin_time: float,
        cos_time: float,
    ) -> TradeSignal:
        """
        Generate trade signal from market context.

        Args:
            context_short: [1, T_short, features] short-term context (always required)
            context_med: [1, T_med, features] medium-term context (None if not available)
            context_long: [1, T_long, features] long-term context (None if not available)
            time_to_close: Normalized time to close (1.0 at open, 0.0 at close)
            sin_time: Sine of time encoding
            cos_time: Cosine of time encoding

        Returns:
            TradeSignal with position, confidence, and exit parameters
        """
        # Compute mask based on time
        mask = self.compute_mask(time_to_close)

        # Encode contexts
        context_short = context_short.to(self.device)
        emb_short = self.short_jepa.encode(context_short)

        if context_med is not None and mask[0, 1] > 0:
            context_med = context_med.to(self.device)
            emb_med = self.med_jepa.encode(context_med)
        else:
            emb_med = torch.zeros_like(emb_short)

        if context_long is not None and mask[0, 2] > 0:
            context_long = context_long.to(self.device)
            emb_long = self.long_jepa.encode(context_long)
        else:
            emb_long = torch.zeros_like(emb_short)

        # Prepare time context
        time_context = torch.tensor(
            [[sin_time, cos_time]],
            dtype=torch.float32,
            device=self.device,
        )

        # Get policy output
        logits, attn_weights = self.policy(
            emb_short, emb_med, emb_long, time_context, mask
        )

        # Get action and confidence
        probs = F.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, action].item()

        # Get dominant scale for exit logic
        dominant_scale, time_limit = self.get_dominant_scale(attn_weights)

        return TradeSignal(
            position=TargetPosition(action),
            confidence=confidence,
            attn_weights=tuple(attn_weights[0].cpu().tolist()),
            dominant_scale=dominant_scale,
            time_limit_mins=time_limit,
        )

    @torch.no_grad()
    def get_signal_from_embeddings(
        self,
        emb_short: torch.Tensor,
        emb_med: torch.Tensor,
        emb_long: torch.Tensor,
        time_to_close: float,
        sin_time: float,
        cos_time: float,
    ) -> TradeSignal:
        """
        Generate trade signal from pre-computed embeddings.

        This is faster if embeddings are already computed.

        Args:
            emb_short: [1, emb_dim] short-term embedding
            emb_med: [1, emb_dim] medium-term embedding (zeros if not available)
            emb_long: [1, emb_dim] long-term embedding (zeros if not available)
            time_to_close: Normalized time to close
            sin_time: Sine of time encoding
            cos_time: Cosine of time encoding

        Returns:
            TradeSignal with position, confidence, and exit parameters
        """
        # Compute mask based on time
        mask = self.compute_mask(time_to_close)

        # Move to device
        emb_short = emb_short.to(self.device)
        emb_med = emb_med.to(self.device) * mask[0, 1]  # Zero out if masked
        emb_long = emb_long.to(self.device) * mask[0, 2]

        # Prepare time context
        time_context = torch.tensor(
            [[sin_time, cos_time]],
            dtype=torch.float32,
            device=self.device,
        )

        # Get policy output
        logits, attn_weights = self.policy(
            emb_short, emb_med, emb_long, time_context, mask
        )

        # Get action and confidence
        probs = F.softmax(logits, dim=-1)
        action = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, action].item()

        # Get dominant scale for exit logic
        dominant_scale, time_limit = self.get_dominant_scale(attn_weights)

        return TradeSignal(
            position=TargetPosition(action),
            confidence=confidence,
            attn_weights=tuple(attn_weights[0].cpu().tolist()),
            dominant_scale=dominant_scale,
            time_limit_mins=time_limit,
        )

    def get_exit_parameters(
        self,
        signal: TradeSignal,
        current_price: float,
        atr: float,
    ) -> ExitParameters:
        """
        Compute exit parameters for a trade.

        Uses the dominant attention weight to determine time limit.

        Args:
            signal: Trade signal from get_signal()
            current_price: Current underlying price
            atr: Average True Range for volatility scaling

        Returns:
            ExitParameters with stop loss, take profit, and time limit
        """
        # Base parameters
        stop_loss_mult = 1.0
        take_profit_mult = 2.0

        # Adjust based on dominant scale
        if signal.dominant_scale == "short":
            # Scalp: tighter stops, quicker exits
            stop_loss_mult = 0.75
            take_profit_mult = 1.5
        elif signal.dominant_scale == "medium":
            # Swing: standard parameters
            stop_loss_mult = 1.0
            take_profit_mult = 2.0
        else:  # long
            # Trend: wider stops, larger targets
            stop_loss_mult = 1.5
            take_profit_mult = 3.0

        return ExitParameters(
            stop_loss_atr_mult=stop_loss_mult,
            take_profit_atr_mult=take_profit_mult,
            time_limit_mins=signal.time_limit_mins,
        )

    def should_execute_trade(
        self,
        signal: TradeSignal,
        min_confidence: float = 0.6,
        current_position: TargetPosition = TargetPosition.NEUTRAL,
    ) -> bool:
        """
        Determine if we should execute a trade based on signal.

        Args:
            signal: Trade signal from get_signal()
            min_confidence: Minimum confidence threshold
            current_position: Current portfolio position

        Returns:
            True if trade should be executed
        """
        # Don't trade if confidence is too low
        if signal.confidence < min_confidence:
            return False

        # Don't trade if signal matches current position
        if signal.position == current_position:
            return False

        # Don't trade neutral signals (they're for closing, not opening)
        if signal.position == TargetPosition.NEUTRAL:
            # Unless we have a position to close
            return current_position != TargetPosition.NEUTRAL

        return True
