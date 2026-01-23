"""
Configuration dataclasses for Multi-Scale Percentile Policy Fusion.

This module defines the configuration for combining multiple independently-trained
percentile policy models (15m, 5m, 1m horizons) into a unified trading system.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class Signal(IntEnum):
    """Trading signal enum."""
    HOLD = 0
    LONG = 1  # BUY_CALL
    SHORT = 2  # BUY_PUT


@dataclass
class PercentileModelConfig:
    """Configuration for a single percentile policy model."""
    name: str                    # Model identifier: "15m", "5m", "1m"
    lejepa_path: str             # Path to LeJEPA checkpoint
    policy_path: str             # Path to policy checkpoint
    context_len: int             # Context length in minutes: 60, 15, 5
    horizon_minutes: int         # Prediction horizon: 15, 5, 1
    long_threshold: float        # Calibrated threshold for LONG signal
    short_threshold: float       # Calibrated threshold for SHORT signal
    base_weight: float           # Base weight in fusion (0.333 for equal)
    min_context_minutes: int     # Minutes after open before model is available

    @property
    def max_hold_minutes(self) -> int:
        """Maximum hold time based on horizon (2x horizon)."""
        # 1m → 2min, 5m → 10min, 15m → 30min
        return self.horizon_minutes * 2


@dataclass
class TradingWindow:
    """Trading window constraints."""
    open_after_minutes: int = 15   # No trades in first 15 min (9:30-9:45)
    close_before_minutes: int = 15  # No trades in last 15 min (3:45-4:00)
    market_open_hour: int = 9
    market_open_minute: int = 30
    market_close_hour: int = 16
    market_close_minute: int = 0

    @property
    def market_minutes(self) -> int:
        """Total minutes in trading day."""
        return (self.market_close_hour * 60 + self.market_close_minute) - \
               (self.market_open_hour * 60 + self.market_open_minute)


@dataclass
class FusedSignal:
    """Output of the fusion algorithm."""
    action: Signal               # Final trading decision: HOLD, LONG, SHORT
    confidence: float            # Weighted confidence score
    dominant_model: str          # Model with highest contribution: "15m", "5m", "1m"
    individual_signals: Dict[str, Tuple[Signal, float]]  # Per-model signals
    exit_horizon_minutes: int    # Exit horizon based on agreeing models
    attention_weights: Tuple[float, float, float]  # Actual weights used (15m, 5m, 1m)
    agreeing_models: Tuple[str, ...]  # Which models agreed on the signal
    position_size_multiplier: float = 1.0  # 2x when all 3 models agree


@dataclass
class FusionConfig:
    """Configuration for the fusion algorithm."""
    # Model weights (must sum to 1.0)
    weights: Tuple[float, float, float] = (0.333, 0.333, 0.333)  # 15m, 5m, 1m

    # Minimum weighted agreement for directional signal
    min_agreement: float = 0.40

    # Trading window
    trading_window: TradingWindow = field(default_factory=TradingWindow)

    # Exit parameters based on dominant model
    stop_loss_pct: float = 10.0  # Stop loss percentage for options

    def __post_init__(self):
        """Validate configuration."""
        if abs(sum(self.weights) - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {sum(self.weights)}")


# Default model configurations based on the plan
# Thresholds calibrated for ~15 signals/day per model on 2023-2025 data
DEFAULT_MODEL_CONFIGS: List[PercentileModelConfig] = [
    PercentileModelConfig(
        name="15m",
        lejepa_path="data/checkpoints/lejepa-60-15/lejepa_best.pt",
        policy_path="data/checkpoints/percentile-60-15-v2/entry_policy_best.pt",
        context_len=60,
        horizon_minutes=15,
        long_threshold=0.5171,  # Calibrated for ~15 LONG signals/day
        short_threshold=0.6009,  # Calibrated for ~15 SHORT signals/day
        base_weight=0.333,
        min_context_minutes=60,
    ),
    PercentileModelConfig(
        name="5m",
        lejepa_path="data/checkpoints/lejepa-15-5-v1/lejepa_best.pt",  # v1 (v2 overfit to HOLD)
        policy_path="data/checkpoints/percentile-15-5-v1/entry_policy_best.pt",  # v1 has balanced predictions
        context_len=15,
        horizon_minutes=5,
        long_threshold=0.4973,  # Calibrated for ~15 LONG signals/day
        short_threshold=0.5534,  # Calibrated for ~15 SHORT signals/day
        base_weight=0.333,
        min_context_minutes=15,
    ),
    PercentileModelConfig(
        name="1m",
        lejepa_path="data/checkpoints/lejepa-5-1-v1/lejepa_best.pt",
        policy_path="data/checkpoints/percentile-5-1-v1/entry_policy_best.pt",
        context_len=5,
        horizon_minutes=1,
        long_threshold=0.5723,  # Calibrated for ~15 LONG signals/day
        short_threshold=0.6837,  # Calibrated for ~15 SHORT signals/day
        base_weight=0.333,
        min_context_minutes=5,
    ),
]


def get_default_configs() -> Tuple[List[PercentileModelConfig], FusionConfig]:
    """Get default model and fusion configurations."""
    return DEFAULT_MODEL_CONFIGS.copy(), FusionConfig()
