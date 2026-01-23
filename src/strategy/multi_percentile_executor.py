"""
Multi-Scale Percentile Policy Executor.

Combines three independently-trained percentile policy models (15m, 5m, 1m horizons)
into a single unified trading system using equal-weight voting fusion.

Architecture:
    Context Data → Data Buffer (rolling 60-min history)
                      │
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
   LeJEPA-60      LeJEPA-15      LeJEPA-5
    (15m)          (5m)           (1m)
       │              │              │
       ▼              ▼              ▼
   Policy-15m    Policy-5m     Policy-1m
       │              │              │
       └──────────────┼──────────────┘
                      ▼
              Fusion Algorithm
              (weighted voting)
                      ▼
               FusedSignal
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.model.lejepa import LeJEPA
from src.model.policy import EntryPolicy
from src.strategy.fusion_config import (
    DEFAULT_MODEL_CONFIGS,
    FusedSignal,
    FusionConfig,
    PercentileModelConfig,
    Signal,
    TradingWindow,
)


@dataclass
class LoadedModel:
    """Container for a loaded LeJEPA + Policy pair."""
    config: PercentileModelConfig
    lejepa: LeJEPA
    policy: EntryPolicy


def _strip_compiled_prefix(state_dict: dict) -> dict:
    """Strip _orig_mod prefix from state dict keys (added by torch.compile)."""
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("._orig_mod", "")
        new_state_dict[new_key] = value
    return new_state_dict


class MultiPercentileExecutor:
    """
    Production executor for multi-scale percentile policy fusion.

    Loads multiple LeJEPA + Policy pairs and fuses their signals using
    equal-weight voting with time-based masking.

    Example usage:
        executor = MultiPercentileExecutor.from_configs(
            model_configs=DEFAULT_MODEL_CONFIGS,
            fusion_config=FusionConfig(),
            device="cuda",
        )

        # Get fused signal
        signal = executor.get_signal(
            contexts={
                "15m": context_60_tensor,  # [1, 60, features]
                "5m": context_15_tensor,   # [1, 15, features]
                "1m": context_5_tensor,    # [1, 5, features]
            },
            minutes_elapsed=45,  # Minutes since market open
        )
    """

    def __init__(
        self,
        models: Dict[str, LoadedModel],
        fusion_config: FusionConfig,
        device: torch.device,
    ):
        """
        Initialize the executor.

        Args:
            models: Dict mapping model name to LoadedModel
            fusion_config: Fusion algorithm configuration
            device: Device for inference
        """
        self.models = models
        self.fusion_config = fusion_config
        self.device = device

        # Model order for consistent indexing
        self.model_order = ["15m", "5m", "1m"]

        # Freeze all models
        for name, loaded in self.models.items():
            loaded.lejepa.eval()
            loaded.policy.eval()
            for param in loaded.lejepa.parameters():
                param.requires_grad = False
            for param in loaded.policy.parameters():
                param.requires_grad = False

    @classmethod
    def from_configs(
        cls,
        model_configs: List[PercentileModelConfig],
        fusion_config: Optional[FusionConfig] = None,
        device: str = "cuda",
    ) -> "MultiPercentileExecutor":
        """
        Create executor from model configurations.

        Args:
            model_configs: List of model configurations
            fusion_config: Fusion algorithm configuration (default: FusionConfig())
            device: Device for inference

        Returns:
            Configured MultiPercentileExecutor
        """
        if fusion_config is None:
            fusion_config = FusionConfig()

        device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
        models = {}

        for config in model_configs:
            lejepa, policy = cls._load_model_pair(
                config.lejepa_path,
                config.policy_path,
                device_obj,
            )
            models[config.name] = LoadedModel(
                config=config,
                lejepa=lejepa,
                policy=policy,
            )
            print(f"Loaded {config.name} model (context={config.context_len}, horizon={config.horizon_minutes})")

        return cls(models, fusion_config, device_obj)

    @staticmethod
    def _load_model_pair(
        lejepa_path: str,
        policy_path: str,
        device: torch.device,
    ) -> Tuple[LeJEPA, EntryPolicy]:
        """Load a LeJEPA + Policy pair from checkpoints."""
        # Load LeJEPA
        lejepa_ckpt = torch.load(lejepa_path, map_location="cpu")
        lejepa_config = lejepa_ckpt["config"]

        lejepa = LeJEPA(
            input_dim=lejepa_config.get("input_dim", 95),
            d_model=lejepa_config.get("d_model", 512),
            nhead=lejepa_config.get("nhead", 8),
            num_layers=lejepa_config.get("num_layers", 6),
            embedding_dim=lejepa_config.get("embedding_dim", 64),
            max_context_len=lejepa_config.get("max_context_len", 60),
            dropout=lejepa_config.get("dropout", 0.1),
            lambda_reg=lejepa_config.get("lambda_reg", 0.5),
            reg_type=lejepa_config.get("reg_type", "sigreg"),
        )
        state_dict = _strip_compiled_prefix(lejepa_ckpt["state_dict"])
        lejepa.load_state_dict(state_dict)
        lejepa = lejepa.to(device)

        # Load policy
        policy_ckpt = torch.load(policy_path, map_location="cpu")
        policy_config = policy_ckpt["config"]

        policy = EntryPolicy(
            embedding_dim=policy_config["embedding_dim"],
            hidden_dim=policy_config["hidden_dim"],
            num_layers=policy_config["num_layers"],
            num_actions=policy_config.get("num_classes", policy_config.get("num_actions", 3)),
        )
        policy.load_state_dict(policy_ckpt["policy_state_dict"])
        policy = policy.to(device)

        return lejepa, policy

    def compute_mask(self, minutes_elapsed: int) -> Dict[str, bool]:
        """
        Compute availability mask based on time elapsed since market open.

        Args:
            minutes_elapsed: Minutes since market open (9:30 AM ET)

        Returns:
            Dict mapping model name to availability (True if available)
        """
        mask = {}
        for name, loaded in self.models.items():
            mask[name] = minutes_elapsed >= loaded.config.min_context_minutes
        return mask

    def is_trading_allowed(self, minutes_elapsed: int) -> bool:
        """
        Check if trading is allowed based on time of day.

        Args:
            minutes_elapsed: Minutes since market open

        Returns:
            True if within trading window
        """
        window = self.fusion_config.trading_window

        # Not in first N minutes
        if minutes_elapsed < window.open_after_minutes:
            return False

        # Not in last N minutes
        time_remaining = window.market_minutes - minutes_elapsed
        if time_remaining < window.close_before_minutes:
            return False

        return True

    @torch.no_grad()
    def get_individual_signal(
        self,
        model_name: str,
        context: torch.Tensor,
    ) -> Tuple[Signal, np.ndarray]:
        """
        Get trading signal from a single model.

        Args:
            model_name: Model identifier ("15m", "5m", "1m")
            context: Context tensor [1, context_len, features]

        Returns:
            Tuple of (Signal, probabilities array)
        """
        loaded = self.models[model_name]
        config = loaded.config

        # Encode context
        context = context.to(self.device)
        embeddings = loaded.lejepa.encode(context)  # [1, embed_dim]

        # Get policy output
        output = loaded.policy(embeddings)
        probs = output["action_probs"][0].cpu().numpy()

        long_prob = probs[Signal.LONG]
        short_prob = probs[Signal.SHORT]

        # Check thresholds
        long_signal = long_prob >= config.long_threshold
        short_signal = short_prob >= config.short_threshold

        if long_signal and short_signal:
            # Both exceed threshold - pick the one further above threshold
            long_excess = (long_prob - config.long_threshold) / config.long_threshold
            short_excess = (short_prob - config.short_threshold) / config.short_threshold
            signal = Signal.LONG if long_excess >= short_excess else Signal.SHORT
        elif long_signal:
            signal = Signal.LONG
        elif short_signal:
            signal = Signal.SHORT
        else:
            signal = Signal.HOLD

        return signal, probs

    def fuse_signals(
        self,
        signals: Dict[str, Tuple[Signal, np.ndarray]],
        mask: Dict[str, bool],
    ) -> FusedSignal:
        """
        Fuse individual model signals using model count-based voting.

        Agreement rules:
        - 2 models available: both must agree (2/2)
        - 3 models available: at least 2 must agree (2/3)

        Args:
            signals: Dict mapping model name to (Signal, probs)
            mask: Dict mapping model name to availability

        Returns:
            FusedSignal with fused action and metadata
        """
        base_weights = self.fusion_config.weights
        model_names = self.model_order

        # Count available models and collect their signals
        available_models = []
        individual_signals = {}
        for i, name in enumerate(model_names):
            if mask.get(name, False) and name in signals:
                available_models.append(name)
                signal, probs = signals[name]
                individual_signals[name] = (signal, float(probs[signal]))

        num_available = len(available_models)

        # Need at least 2 models to trade
        if num_available < 2:
            return FusedSignal(
                action=Signal.HOLD,
                confidence=0.0,
                dominant_model="",
                individual_signals=individual_signals,
                exit_horizon_minutes=0,
                attention_weights=(0.0, 0.0, 0.0),
                agreeing_models=(),
            )

        # Count votes for each direction
        long_count = 0
        short_count = 0
        hold_count = 0
        long_confidence = 0.0
        short_confidence = 0.0

        for name in available_models:
            signal, prob = individual_signals[name]
            if signal == Signal.LONG:
                long_count += 1
                long_confidence += prob
            elif signal == Signal.SHORT:
                short_count += 1
                short_confidence += prob
            else:
                hold_count += 1

        # Determine required agreement based on number of models
        # 2 models: need 2/2 (both agree)
        # 3 models: need 2/3 (majority)
        required_agreement = 2

        # Calculate actual weights for output (normalized)
        actual_weights = []
        for i, name in enumerate(model_names):
            if name in available_models:
                actual_weights.append(base_weights[i])
            else:
                actual_weights.append(0.0)
        weight_sum = sum(actual_weights)
        if weight_sum > 0:
            actual_weights = [w / weight_sum for w in actual_weights]

        # Determine final action based on model count agreement
        # Also track which models agreed
        agreeing_models = []
        if long_count >= required_agreement and long_count > short_count:
            action = Signal.LONG
            confidence = long_confidence / long_count  # Average confidence
            for name in available_models:
                signal, _ = individual_signals[name]
                if signal == Signal.LONG:
                    agreeing_models.append(name)
        elif short_count >= required_agreement and short_count > long_count:
            action = Signal.SHORT
            confidence = short_confidence / short_count
            for name in available_models:
                signal, _ = individual_signals[name]
                if signal == Signal.SHORT:
                    agreeing_models.append(name)
        else:
            action = Signal.HOLD
            confidence = 0.0

        # Determine dominant model (highest confidence among agreeing models)
        dominant_model = ""
        max_prob = 0.0

        for name in agreeing_models:
            signal, prob = individual_signals[name]
            if prob > max_prob:
                max_prob = prob
                dominant_model = name

        # Compute exit horizon based on which models agreed
        # Rules: 1m+5m → 10m, 5m+15m → 15m, 1m+15m → 10m, all 3 → 30m
        exit_horizon = self._compute_exit_horizon(agreeing_models)

        # Triple confluence: all 3 models agree → 2x position size
        triple_confluence = len(agreeing_models) == 3
        position_multiplier = 2.0 if triple_confluence else 1.0

        return FusedSignal(
            action=action,
            confidence=confidence,
            dominant_model=dominant_model,
            individual_signals=individual_signals,
            exit_horizon_minutes=exit_horizon,
            attention_weights=tuple(actual_weights),
            agreeing_models=tuple(agreeing_models),
            position_size_multiplier=position_multiplier,
        )

    def _compute_exit_horizon(self, agreeing_models: List[str]) -> int:
        """
        Compute exit horizon based on which models agreed.

        Rules:
        - 1m + 5m agree → 10 min
        - 5m + 15m agree → 15 min
        - 1m + 15m agree → 10 min
        - All 3 agree → 30 min (triple confluence bonus)
        """
        if not agreeing_models:
            return 0

        has_1m = "1m" in agreeing_models
        has_5m = "5m" in agreeing_models
        has_15m = "15m" in agreeing_models

        if has_1m and has_5m and has_15m:
            return 30  # All 3 agree - triple confluence bonus
        elif has_5m and has_15m:
            return 15  # 5m + 15m
        elif has_1m and has_5m:
            return 10  # 1m + 5m
        elif has_1m and has_15m:
            return 10  # 1m + 15m
        else:
            return 0  # Should not happen with required_agreement = 2

    @torch.no_grad()
    def get_signal(
        self,
        contexts: Dict[str, torch.Tensor],
        minutes_elapsed: int,
    ) -> FusedSignal:
        """
        Get fused trading signal from all available models.

        Args:
            contexts: Dict mapping model name to context tensor
                - "15m": [1, 60, features]
                - "5m": [1, 15, features]
                - "1m": [1, 5, features]
            minutes_elapsed: Minutes since market open (9:30 AM ET)

        Returns:
            FusedSignal with fused action and metadata
        """
        # Compute availability mask
        mask = self.compute_mask(minutes_elapsed)

        # Get individual signals from available models
        signals = {}
        for name in self.model_order:
            if mask.get(name, False) and name in contexts:
                context = contexts[name]
                signals[name] = self.get_individual_signal(name, context)

        # Fuse signals
        return self.fuse_signals(signals, mask)

    def get_model_info(self) -> Dict[str, Dict]:
        """Get information about loaded models."""
        info = {}
        for name, loaded in self.models.items():
            info[name] = {
                "context_len": loaded.config.context_len,
                "horizon_minutes": loaded.config.horizon_minutes,
                "long_threshold": loaded.config.long_threshold,
                "short_threshold": loaded.config.short_threshold,
                "min_context_minutes": loaded.config.min_context_minutes,
                "max_hold_minutes": loaded.config.max_hold_minutes,
            }
        return info
