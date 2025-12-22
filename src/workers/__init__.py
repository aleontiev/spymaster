"""
Workers package for real-time trading system.

This package provides the worker pool architecture for real-time trading:

- EventBus: Async pub/sub for worker coordination
- Syncer: Real-time data synchronization from ThetaData/Polygon
- LoaderWorker: Cache loading and feature computation
- ModelRegistry: Model loading and management
- GraphExecutor: DAG-based model execution
- StrategyRunner: Strategy execution and position management
- WorkerPool: Orchestrates all workers

Usage:
    from src.workers import WorkerPool, load_config

    config = load_config("config/worker_pool.yaml")
    pool = WorkerPool(config)
    await pool.run()
"""

from src.workers.event_bus import EventBus, Event, get_event_bus, reset_event_bus
from src.workers.syncer import Syncer, SyncResult
from src.workers.loader_worker import LoaderWorker, load_minute_for_strategy
from src.workers.model_registry import (
    ModelRegistry,
    LoadedModel,
    ModelType,
    combine_actions,
    combine_exit_actions,
)
from src.workers.graph_executor import (
    ModelGraphExecutor,
    GraphNode,
    create_simple_graph,
    create_multi_timeframe_graph,
)
from src.workers.strategy_runner import (
    StrategyRunner,
    StrategyConfig,
    StrategyAction,
    PositionState,
    ExitMode,
)
from src.workers.pool import (
    WorkerPool,
    WorkerPoolConfig,
    load_config,
    run_worker_pool,
    create_default_config,
)

__all__ = [
    # Event Bus
    "EventBus",
    "Event",
    "get_event_bus",
    "reset_event_bus",
    # Syncer
    "Syncer",
    "SyncResult",
    # Loader
    "LoaderWorker",
    "load_minute_for_strategy",
    # Model Registry
    "ModelRegistry",
    "LoadedModel",
    "ModelType",
    "combine_actions",
    "combine_exit_actions",
    # Graph Executor
    "ModelGraphExecutor",
    "GraphNode",
    "create_simple_graph",
    "create_multi_timeframe_graph",
    # Strategy Runner
    "StrategyRunner",
    "StrategyConfig",
    "StrategyAction",
    "PositionState",
    "ExitMode",
    # Worker Pool
    "WorkerPool",
    "WorkerPoolConfig",
    "load_config",
    "run_worker_pool",
    "create_default_config",
]
