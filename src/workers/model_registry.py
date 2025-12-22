"""
Model Registry for managing loaded models in the worker pool.

Responsible for:
- Loading different model types (lejepa, entry_policy, exit_policy, combiner)
- Sharing models across strategies
- Managing model configurations
- Handling device placement (CPU/GPU)

Models are loaded once and shared to avoid memory duplication.
"""
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from src.model.lejepa import LeJEPA
from src.model.policy import (
    EntryPolicy,
    RegressionEntryPolicy,
    ExitPolicy,
    RuleBasedExitPolicy,
    ContinuousSignalExitPolicy,
    EntryAction,
    ExitAction,
)

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    """Supported model types."""
    LEJEPA = "lejepa"
    ENTRY_POLICY = "entry_policy"
    REGRESSION_POLICY = "regression_policy"
    EXIT_POLICY = "exit_policy"
    RULE_BASED_EXIT = "rule_based_exit"
    CONTINUOUS_EXIT = "continuous_exit"
    COMBINER = "combiner"


@dataclass
class LoadedModel:
    """Container for a loaded model with metadata."""

    id: str
    type: ModelType
    model: Optional[nn.Module]  # None for rule-based/combiner types
    config: Dict[str, Any]
    device: str = "cpu"
    timeframe: str = "1m"  # For LeJEPA: what timeframe this model expects

    def to(self, device: str) -> "LoadedModel":
        """Move model to specified device."""
        if self.model is not None and hasattr(self.model, "to"):
            self.model.to(device)
            self.device = device
        return self

    def eval(self) -> "LoadedModel":
        """Set model to evaluation mode."""
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        return self

    def forward(self, *args, **kwargs) -> Any:
        """Execute the model forward pass."""
        if self.model is None:
            raise ValueError(f"Model {self.id} has no forward method (type: {self.type})")
        return self.model(*args, **kwargs)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension if applicable."""
        if self.type == ModelType.LEJEPA:
            return self.config.get("embedding_dim", 512)
        return 0


@dataclass
class CombinerConfig:
    """Configuration for a combiner node."""
    method: str = "majority"  # majority, weighted_avg, concat
    weights: Optional[List[float]] = None


class ModelRegistry:
    """
    Central registry for all model types.

    Models are loaded once and shared across strategies to avoid memory duplication.

    Supported types:
    - lejepa: LeJEPA encoder (produces embeddings)
    - entry_policy: EntryPolicy network (produces action + confidence)
    - regression_policy: RegressionEntryPolicy (produces continuous score)
    - exit_policy: ExitPolicy network (produces exit decisions)
    - rule_based_exit: RuleBasedExitPolicy (non-neural)
    - continuous_exit: ContinuousSignalExitPolicy (non-neural)
    - combiner: Non-neural combiner (voting, averaging)
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        default_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize the model registry.

        Args:
            device: Default device for models (cuda/cpu)
            default_dtype: Default tensor dtype
        """
        self.device = device
        self.default_dtype = default_dtype
        self.models: Dict[str, LoadedModel] = {}

        # Cache for resampled data (per timeframe)
        self._resample_cache: Dict[str, pd.DataFrame] = {}

    def load_from_config(self, models_config: List[Dict[str, Any]]) -> None:
        """
        Load all models from configuration.

        Args:
            models_config: List of model configurations, each containing:
                - id: Unique model identifier
                - type: Model type (lejepa, entry_policy, etc.)
                - checkpoint: Path to model checkpoint (for neural models)
                - config: Model-specific configuration
        """
        for config in models_config:
            model_id = config["id"]
            model_type = ModelType(config["type"])
            checkpoint_path = config.get("checkpoint")
            model_config = config.get("config", {})

            logger.info(f"Loading model: {model_id} (type: {model_type})")

            try:
                loaded = self._load_model(
                    model_id=model_id,
                    model_type=model_type,
                    checkpoint_path=checkpoint_path,
                    config=model_config,
                )
                self.models[model_id] = loaded
                logger.info(f"Loaded model: {model_id}")

            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                raise

    def _load_model(
        self,
        model_id: str,
        model_type: ModelType,
        checkpoint_path: Optional[str],
        config: Dict[str, Any],
    ) -> LoadedModel:
        """Load a single model based on type."""

        if model_type == ModelType.LEJEPA:
            return self._load_lejepa(model_id, checkpoint_path, config)

        elif model_type == ModelType.ENTRY_POLICY:
            return self._load_entry_policy(model_id, checkpoint_path, config)

        elif model_type == ModelType.REGRESSION_POLICY:
            return self._load_regression_policy(model_id, checkpoint_path, config)

        elif model_type == ModelType.EXIT_POLICY:
            return self._load_exit_policy(model_id, checkpoint_path, config)

        elif model_type == ModelType.RULE_BASED_EXIT:
            return self._load_rule_based_exit(model_id, config)

        elif model_type == ModelType.CONTINUOUS_EXIT:
            return self._load_continuous_exit(model_id, config)

        elif model_type == ModelType.COMBINER:
            return self._load_combiner(model_id, config)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _load_lejepa(
        self,
        model_id: str,
        checkpoint_path: Optional[str],
        config: Dict[str, Any],
    ) -> LoadedModel:
        """Load a LeJEPA encoder."""
        model = LeJEPA(
            input_dim=config.get("input_dim", 34),  # Default for GEX flow features
            d_model=config.get("d_model", 256),
            nhead=config.get("nhead", 8),
            num_layers=config.get("num_layers", 6),
            embedding_dim=config.get("embedding_dim", 512),
            max_seq_len=config.get("max_seq_len", 512),
            dropout=config.get("dropout", 0.1),
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            # Handle different checkpoint formats
            if "context_encoder" in state_dict:
                model.load_state_dict(state_dict["context_encoder"], strict=False)
            elif "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            logger.debug(f"Loaded LeJEPA checkpoint: {checkpoint_path}")

        model = model.to(self.device)
        model.eval()

        return LoadedModel(
            id=model_id,
            type=ModelType.LEJEPA,
            model=model,
            config=config,
            device=self.device,
            timeframe=config.get("timeframe", "1m"),
        )

    def _load_entry_policy(
        self,
        model_id: str,
        checkpoint_path: Optional[str],
        config: Dict[str, Any],
    ) -> LoadedModel:
        """Load an EntryPolicy network."""
        model = EntryPolicy(
            embedding_dim=config.get("input_dim", config.get("embedding_dim", 512)),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            num_actions=config.get("num_classes", config.get("num_actions", 5)),
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            logger.debug(f"Loaded EntryPolicy checkpoint: {checkpoint_path}")

        model = model.to(self.device)
        model.eval()

        return LoadedModel(
            id=model_id,
            type=ModelType.ENTRY_POLICY,
            model=model,
            config=config,
            device=self.device,
        )

    def _load_regression_policy(
        self,
        model_id: str,
        checkpoint_path: Optional[str],
        config: Dict[str, Any],
    ) -> LoadedModel:
        """Load a RegressionEntryPolicy network."""
        model = RegressionEntryPolicy(
            embedding_dim=config.get("input_dim", config.get("embedding_dim", 512)),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            logger.debug(f"Loaded RegressionEntryPolicy checkpoint: {checkpoint_path}")

        model = model.to(self.device)
        model.eval()

        return LoadedModel(
            id=model_id,
            type=ModelType.REGRESSION_POLICY,
            model=model,
            config=config,
            device=self.device,
        )

    def _load_exit_policy(
        self,
        model_id: str,
        checkpoint_path: Optional[str],
        config: Dict[str, Any],
    ) -> LoadedModel:
        """Load an ExitPolicy network."""
        model = ExitPolicy(
            embedding_dim=config.get("input_dim", config.get("embedding_dim", 512)),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            context_dim=config.get("context_dim", 4),
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
            logger.debug(f"Loaded ExitPolicy checkpoint: {checkpoint_path}")

        model = model.to(self.device)
        model.eval()

        return LoadedModel(
            id=model_id,
            type=ModelType.EXIT_POLICY,
            model=model,
            config=config,
            device=self.device,
        )

    def _load_rule_based_exit(
        self,
        model_id: str,
        config: Dict[str, Any],
    ) -> LoadedModel:
        """Load a RuleBasedExitPolicy (non-neural)."""
        policy = RuleBasedExitPolicy(
            take_profit_pct=config.get("take_profit_pct", 50.0),
            risk_reward_ratio=config.get("risk_reward_ratio", 2.0),
            stop_loss_pct=config.get("stop_loss_pct"),
            time_stop_hours=config.get("time_stop_hours"),
            eod_exit_minutes=config.get("eod_exit_minutes", 5.0),
        )

        return LoadedModel(
            id=model_id,
            type=ModelType.RULE_BASED_EXIT,
            model=policy,  # Not a nn.Module but has forward method
            config=config,
            device="cpu",  # Rule-based doesn't use GPU
        )

    def _load_continuous_exit(
        self,
        model_id: str,
        config: Dict[str, Any],
    ) -> LoadedModel:
        """Load a ContinuousSignalExitPolicy (non-neural)."""
        policy = ContinuousSignalExitPolicy(
            counter_signal_confirmation=config.get("counter_signal_confirmation", 2),
            plateau_window=config.get("plateau_window", 15),
            plateau_ratio=config.get("plateau_ratio", 0.5),
            stop_loss_pct=config.get("stop_loss_pct", 25.0),
            take_profit_pct=config.get("take_profit_pct"),
            eod_exit_minutes=config.get("eod_exit_minutes", 5.0),
            flip_on_counter=config.get("flip_on_counter", True),
        )

        return LoadedModel(
            id=model_id,
            type=ModelType.CONTINUOUS_EXIT,
            model=policy,  # Not a nn.Module but has evaluate method
            config=config,
            device="cpu",
        )

    def _load_combiner(
        self,
        model_id: str,
        config: Dict[str, Any],
    ) -> LoadedModel:
        """Load a combiner configuration (non-neural)."""
        return LoadedModel(
            id=model_id,
            type=ModelType.COMBINER,
            model=None,  # Combiners don't have a model
            config=config,
            device="cpu",
        )

    def get(self, model_id: str) -> LoadedModel:
        """
        Get a loaded model by ID.

        Args:
            model_id: Model identifier

        Returns:
            LoadedModel instance

        Raises:
            KeyError: If model not found
        """
        if model_id not in self.models:
            raise KeyError(f"Model not found: {model_id}. Available: {list(self.models.keys())}")
        return self.models[model_id]

    def has(self, model_id: str) -> bool:
        """Check if a model is registered."""
        return model_id in self.models

    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self.models.keys())

    def get_resampled_data(
        self,
        df: pd.DataFrame,
        timeframe: str,
        cache_key: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get resampled data for a timeframe.

        Caches resampled data to avoid redundant computation when
        multiple nodes use the same timeframe.

        Args:
            df: Input DataFrame with 1-minute data
            timeframe: Target timeframe (1m, 5m, 15m, 1h, 1d)
            cache_key: Optional cache key for the resampled data

        Returns:
            Resampled DataFrame
        """
        if timeframe == "1m":
            return df  # No resampling needed

        # Use cache if available
        if cache_key:
            full_key = f"{cache_key}_{timeframe}"
            if full_key in self._resample_cache:
                return self._resample_cache[full_key]

        # Map timeframe to pandas resample rule
        resample_rules = {
            "5m": "5T",
            "15m": "15T",
            "30m": "30T",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }

        rule = resample_rules.get(timeframe)
        if rule is None:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        # Resample the data
        ohlc_cols = ["open", "high", "low", "close"]
        other_cols = [c for c in df.columns if c not in ohlc_cols + ["volume"]]

        resampled = pd.DataFrame()

        # OHLC aggregation
        if all(c in df.columns for c in ohlc_cols):
            ohlc_agg = df[ohlc_cols].resample(rule).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            })
            resampled = ohlc_agg

        # Volume sum
        if "volume" in df.columns:
            resampled["volume"] = df["volume"].resample(rule).sum()

        # Other columns - take last value
        for col in other_cols:
            if col in df.columns:
                resampled[col] = df[col].resample(rule).last()

        # Forward fill NaNs
        resampled = resampled.ffill().dropna()

        # Cache the result
        if cache_key:
            full_key = f"{cache_key}_{timeframe}"
            self._resample_cache[full_key] = resampled

        return resampled

    def clear_resample_cache(self) -> None:
        """Clear the resampled data cache."""
        self._resample_cache.clear()

    def unload_model(self, model_id: str) -> None:
        """Unload a model from the registry."""
        if model_id in self.models:
            model = self.models.pop(model_id)
            if model.model is not None:
                del model.model
            logger.info(f"Unloaded model: {model_id}")

    def unload_all(self) -> None:
        """Unload all models."""
        for model_id in list(self.models.keys()):
            self.unload_model(model_id)
        self._resample_cache.clear()


def combine_actions(
    actions: List[EntryAction],
    method: str = "majority",
    weights: Optional[List[float]] = None,
) -> EntryAction:
    """
    Combine multiple entry actions using the specified method.

    Args:
        actions: List of EntryAction values
        method: Combination method (majority, weighted_avg)
        weights: Optional weights for weighted_avg

    Returns:
        Combined EntryAction
    """
    if not actions:
        return EntryAction.HOLD

    if method == "majority":
        # Simple majority vote
        from collections import Counter
        counts = Counter(actions)
        return counts.most_common(1)[0][0]

    elif method == "weighted_avg":
        # Weight the action indices
        if weights is None:
            weights = [1.0] * len(actions)

        weighted_sum = sum(
            action.value * weight
            for action, weight in zip(actions, weights)
        )
        total_weight = sum(weights)
        avg = weighted_sum / total_weight if total_weight > 0 else 0

        # Round to nearest action
        idx = int(round(avg))
        idx = max(0, min(idx, len(EntryAction) - 1))
        return EntryAction(idx)

    else:
        raise ValueError(f"Unknown combination method: {method}")


def combine_exit_actions(
    actions: List[ExitAction],
    method: str = "conservative",
) -> ExitAction:
    """
    Combine multiple exit actions.

    Args:
        actions: List of ExitAction values
        method: Combination method (conservative = any CLOSE triggers close)

    Returns:
        Combined ExitAction
    """
    if not actions:
        return ExitAction.HOLD_POSITION

    if method == "conservative":
        # If any model says CLOSE, close
        if ExitAction.CLOSE in actions:
            return ExitAction.CLOSE
        return ExitAction.HOLD_POSITION

    elif method == "majority":
        from collections import Counter
        counts = Counter(actions)
        return counts.most_common(1)[0][0]

    else:
        raise ValueError(f"Unknown combination method: {method}")
