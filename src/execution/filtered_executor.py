"""
Filtered executor for model combinations.

Wraps MultiPercentileExecutor to filter signals to specific model combinations.
"""
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

from src.strategy.fusion_config import FusedSignal, Signal
from src.strategy.multi_percentile_executor import MultiPercentileExecutor


class FilteredExecutor:
    """
    Wrapper that filters signals to only consider specific model combinations.
    """

    def __init__(
        self,
        executor: MultiPercentileExecutor,
        allowed_models: Set[str],
    ):
        self.executor = executor
        self.allowed_models = allowed_models
        self.fusion_config = executor.fusion_config

    def is_trading_allowed(self, minutes_elapsed: int) -> bool:
        return self.executor.is_trading_allowed(minutes_elapsed)

    def get_model_info(self) -> Dict:
        return {k: v for k, v in self.executor.get_model_info().items()
                if k in self.allowed_models}

    def get_signal(
        self,
        contexts: Dict[str, torch.Tensor],
        minutes_elapsed: int,
    ) -> FusedSignal:
        """Get fused signal considering only allowed models."""
        # Filter contexts to only allowed models
        filtered_contexts = {k: v for k, v in contexts.items() if k in self.allowed_models}

        # Get individual signals
        mask = self.executor.compute_mask(minutes_elapsed)
        filtered_mask = {k: v for k, v in mask.items() if k in self.allowed_models}

        signals = {}
        for name in self.allowed_models:
            if filtered_mask.get(name, False) and name in filtered_contexts:
                context = filtered_contexts[name]
                signals[name] = self.executor.get_individual_signal(name, context)

        # Custom fusion for filtered models
        return self._fuse_filtered_signals(signals, filtered_mask)

    def _fuse_filtered_signals(
        self,
        signals: Dict[str, Tuple[Signal, np.ndarray]],
        mask: Dict[str, bool],
    ) -> FusedSignal:
        """Fuse signals from filtered models only.

        Trading logic:
        - For 2-model combinations: need both to agree (2/2)
        - For 3-model combinations: need ALL 3 to agree (3/3)
        """
        model_order = [m for m in ["15m", "5m", "1m"] if m in self.allowed_models]

        available_models = []
        individual_signals = {}
        for name in model_order:
            if mask.get(name, False) and name in signals:
                available_models.append(name)
                signal, probs = signals[name]
                individual_signals[name] = (signal, float(probs[signal]))

        num_available = len(available_models)
        num_required = len(self.allowed_models)

        # Need all allowed models to be available
        if num_available < num_required:
            return FusedSignal(
                action=Signal.HOLD,
                confidence=0.0,
                dominant_model="",
                individual_signals=individual_signals,
                exit_horizon_minutes=0,
                attention_weights=(0.0, 0.0, 0.0),
                agreeing_models=(),
            )

        # Count votes
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

        # For 2 models: need both to agree (2/2)
        # For 3 models: need ALL 3 to agree (3/3) - changed from 2/3
        required_agreement = num_required  # All must agree

        agreeing_models = []
        action = Signal.HOLD
        confidence = 0.0

        # Check for valid LONG signal - ALL models must agree
        if long_count == required_agreement:
            action = Signal.LONG
            confidence = long_confidence / long_count
            for name in available_models:
                signal, _ = individual_signals[name]
                if signal == Signal.LONG:
                    agreeing_models.append(name)
        # Check for valid SHORT signal - ALL models must agree
        elif short_count == required_agreement:
            action = Signal.SHORT
            confidence = short_confidence / short_count
            for name in available_models:
                signal, _ = individual_signals[name]
                if signal == Signal.SHORT:
                    agreeing_models.append(name)

        # Dominant model
        dominant_model = ""
        max_prob = 0.0
        for name in agreeing_models:
            signal, prob = individual_signals[name]
            if prob > max_prob:
                max_prob = prob
                dominant_model = name

        # Exit horizon
        exit_horizon = self._compute_exit_horizon(agreeing_models)

        return FusedSignal(
            action=action,
            confidence=confidence,
            dominant_model=dominant_model,
            individual_signals=individual_signals,
            exit_horizon_minutes=exit_horizon,
            attention_weights=(0.333, 0.333, 0.333),
            agreeing_models=tuple(agreeing_models),
            position_size_multiplier=1.0,  # Now using confidence-based scaling instead
        )

    def _compute_exit_horizon(self, agreeing_models: List[str]) -> int:
        """Compute exit horizon based on agreeing models (reduced by 2x for tighter exits)."""
        if not agreeing_models:
            return 0

        has_1m = "1m" in agreeing_models
        has_5m = "5m" in agreeing_models
        has_15m = "15m" in agreeing_models

        # Reduced time barriers by 2x (runners handle extended moves now)
        if has_1m and has_5m and has_15m:
            return 30   # was 60
        elif has_5m and has_15m:
            return 15   # was 30
        elif has_1m and has_5m:
            return 10   # was 20
        elif has_1m and has_15m:
            return 10   # was 20
        else:
            return 0
