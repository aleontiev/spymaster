"""
Worker Pool for orchestrating all workers.

Responsible for:
- Loading configuration from YAML
- Initializing all workers with shared resources
- Managing worker lifecycle (start, stop, health)
- Coordinating between workers via event bus

Usage:
    config = load_config("config/worker_pool.yaml")
    pool = WorkerPool(config)
    await pool.run()
"""
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
import torch

from src.workers.event_bus import EventBus
from src.workers.syncer import Syncer
from src.workers.loader_worker import LoaderWorker
from src.workers.strategy_runner import StrategyRunner, StrategyConfig
from src.workers.model_registry import ModelRegistry
from src.workers.graph_executor import GraphNode

logger = logging.getLogger(__name__)


@dataclass
class WorkerPoolConfig:
    """Configuration for the worker pool."""

    underlying: str = "SPY"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Data source configuration
    thetadata: Dict[str, Any] = field(default_factory=lambda: {
        "terminal_url": "http://localhost:25503/v3",
        "timeout": 60,
    })
    polygon: Dict[str, Any] = field(default_factory=lambda: {
        "delay_minutes": 0,  # Pro plan = real-time
    })

    # Model definitions
    models: List[Dict[str, Any]] = field(default_factory=list)

    # Strategy definitions
    strategies: List[Dict[str, Any]] = field(default_factory=list)

    # Worker settings
    syncer_delay_minutes: int = 0
    syncer_strike_range: float = 20.0

    # Cache directories
    stocks_dir: Optional[str] = None
    options_dir: Optional[str] = None
    gex_flow_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    raw_cache_dir: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkerPoolConfig":
        """Create config from dictionary."""
        return cls(
            underlying=d.get("underlying", "SPY"),
            device=d.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            thetadata=d.get("thetadata", {}),
            polygon=d.get("polygon", {}),
            models=d.get("models", []),
            strategies=d.get("strategies", []),
            syncer_delay_minutes=d.get("syncer_delay_minutes", 0),
            syncer_strike_range=d.get("syncer_strike_range", 20.0),
            stocks_dir=d.get("stocks_dir"),
            options_dir=d.get("options_dir"),
            gex_flow_dir=d.get("gex_flow_dir"),
            cache_dir=d.get("cache_dir"),
            raw_cache_dir=d.get("raw_cache_dir"),
        )


def load_config(path: str) -> WorkerPoolConfig:
    """
    Load worker pool configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        WorkerPoolConfig instance
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return WorkerPoolConfig.from_dict(data)


class WorkerPool:
    """
    Manages the worker pool and coordinates between workers.

    Workers:
    - EventBus: Async pub/sub for worker coordination
    - Syncer: Real-time data synchronization
    - LoaderWorker: Cache loading and feature computation
    - StrategyRunner: Strategy execution and decisions

    Usage:
        config = load_config("config/worker_pool.yaml")
        pool = WorkerPool(config)
        await pool.run()
    """

    def __init__(self, config: WorkerPoolConfig) -> None:
        """
        Initialize the worker pool.

        Args:
            config: Worker pool configuration
        """
        self.config = config
        self._running = False

        # Initialize event bus
        self.event_bus = EventBus()

        # Initialize model registry
        logger.info("Initializing model registry...")
        self.model_registry = ModelRegistry(device=config.device)

        if config.models:
            # Add rule-based exit models that strategies might reference
            self._add_rule_based_exits(config)
            self.model_registry.load_from_config(config.models)
            logger.info(f"Loaded {len(self.model_registry.list_models())} models")

        # Parse strategy configs
        strategy_configs = []
        for strat_dict in config.strategies:
            strategy_configs.append(StrategyConfig.from_dict(strat_dict))
        logger.info(f"Loaded {len(strategy_configs)} strategy configurations")

        # Initialize workers
        logger.info("Initializing workers...")

        # Syncer
        self.syncer = Syncer(
            underlying=config.underlying,
            delay_minutes=config.syncer_delay_minutes,
            strike_range=config.syncer_strike_range,
            thetadata_config=config.thetadata,
            polygon_config=config.polygon,
            event_bus=self.event_bus,
        )

        # Loader
        self.loader = LoaderWorker(
            underlying=config.underlying,
            event_bus=self.event_bus,
            stocks_dir=Path(config.stocks_dir) if config.stocks_dir else None,
            options_dir=Path(config.options_dir) if config.options_dir else None,
            gex_flow_dir=Path(config.gex_flow_dir) if config.gex_flow_dir else None,
            cache_dir=Path(config.cache_dir) if config.cache_dir else None,
            raw_cache_dir=Path(config.raw_cache_dir) if config.raw_cache_dir else None,
        )

        # Strategy runner
        self.strategy_runner = StrategyRunner(
            strategies=strategy_configs,
            model_registry=self.model_registry,
            event_bus=self.event_bus,
            cache_dir=Path(config.cache_dir) if config.cache_dir else None,
            underlying=config.underlying,
            device=config.device,
        )

        logger.info("Worker pool initialized")

    def _add_rule_based_exits(self, config: WorkerPoolConfig) -> None:
        """
        Add rule-based exit models that strategies might reference.

        These are registered in the model registry so they can be used
        in model graphs just like neural models.
        """
        # Check if any strategy uses rule-based or continuous exit
        for strat in config.strategies:
            exit_mode = strat.get("exit_mode", "rule_based")

            if exit_mode == "rule_based":
                # Check if a rule_exit model is referenced but not defined
                has_rule_exit = any(
                    m.get("id") == "rule_exit"
                    for m in config.models
                )
                if not has_rule_exit:
                    config.models.append({
                        "id": "rule_exit",
                        "type": "rule_based_exit",
                        "config": {
                            "take_profit_pct": strat.get("take_profit_pct", 50.0),
                            "risk_reward_ratio": strat.get("risk_reward_ratio", 2.0),
                            "stop_loss_pct": strat.get("stop_loss_pct"),
                        },
                    })

            elif exit_mode == "continuous_signal":
                has_continuous_exit = any(
                    m.get("id") == "continuous_exit"
                    for m in config.models
                )
                if not has_continuous_exit:
                    config.models.append({
                        "id": "continuous_exit",
                        "type": "continuous_exit",
                        "config": {
                            "counter_signal_confirmation": strat.get("counter_signal_confirmation", 2),
                            "plateau_window": strat.get("plateau_window", 15),
                            "plateau_ratio": strat.get("plateau_ratio", 0.5),
                            "stop_loss_pct": strat.get("stop_loss_pct", 25.0),
                        },
                    })

    async def run(self) -> None:
        """
        Start all workers and run until stopped.

        Workers are started concurrently:
        - Event bus processing
        - Syncer (minute-level REST polling)
        - Loader (triggered by sync events)
        - StrategyRunner (triggered by cache events)
        """
        logger.info("Starting worker pool...")
        self._running = True

        try:
            # Start all workers concurrently
            await asyncio.gather(
                self.event_bus.run(),
                self.syncer.run(),
                self.loader.run(),
                self.strategy_runner.run(),
            )

        except asyncio.CancelledError:
            logger.info("Worker pool cancelled")

        finally:
            self.stop()
            logger.info("Worker pool stopped")

    def stop(self) -> None:
        """Graceful shutdown of all workers."""
        logger.info("Stopping worker pool...")
        self._running = False

        # Stop workers in order
        self.syncer.stop()
        self.loader.stop()
        self.strategy_runner.stop()
        self.event_bus.stop()

        # Cleanup model registry
        self.model_registry.unload_all()

    @property
    def is_running(self) -> bool:
        """Check if the pool is running."""
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """Get status of all workers."""
        return {
            "event_bus": self.event_bus.is_running,
            "syncer": self.syncer.is_running,
            "loader": self.loader.is_running,
            "strategy_runner": self.strategy_runner.is_running,
            "models_loaded": len(self.model_registry.list_models()),
            "strategies": [s.name for s in self.strategy_runner.strategies],
        }


async def run_worker_pool(config_path: str) -> None:
    """
    Convenience function to run the worker pool from a config file.

    Args:
        config_path: Path to YAML configuration file
    """
    config = load_config(config_path)
    pool = WorkerPool(config)

    try:
        await pool.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        pool.stop()


# Default configuration for quick start
DEFAULT_CONFIG = """
# Worker Pool Configuration
underlying: SPY
device: cuda

thetadata:
  terminal_url: http://localhost:25503/v3
  timeout: 60

polygon:
  delay_minutes: 0

# Model definitions (loaded once, shared across strategies)
models:
  - id: jepa_1m
    type: lejepa
    checkpoint: checkpoints/lejepa_1m.pt
    config:
      input_dim: 34
      d_model: 256
      nhead: 8
      num_layers: 6
      embedding_dim: 512
      patch_length: 64
      timeframe: 1m

  - id: policy_5class
    type: entry_policy
    checkpoint: checkpoints/policy_5class.pt
    config:
      input_dim: 512
      num_classes: 5

# Strategy definitions
strategies:
  - name: momentum_v1
    entry_confidence: 0.4
    exit_mode: rule_based
    take_profit_pct: 50.0
    risk_reward_ratio: 2.0
    patch_length: 64
    model_graph:
      - node: jepa_1m
        inputs: [market_data]
        outputs: [embedding]
      - node: policy_5class
        inputs: [embedding]
        outputs: [entry_action]
      - node: rule_exit
        type: rule_based_exit
        inputs: [position_context]
        outputs: [exit_action]
        config:
          take_profit_pct: 50.0
          risk_reward_ratio: 2.0
"""


def create_default_config(output_path: str = "config/worker_pool.yaml") -> None:
    """
    Create a default configuration file.

    Args:
        output_path: Path to write configuration file
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        f.write(DEFAULT_CONFIG)

    logger.info(f"Created default config at {output_path}")
