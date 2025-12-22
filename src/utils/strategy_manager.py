"""
Strategy Manager for Spymaster.

Manages trading strategies that combine model checkpoints with backtest/execution configuration.

A strategy consists of:
- UUID id for database-like identification
- LeJEPA checkpoint for market embeddings (with its own UUID)
- Entry policy checkpoint for trade entry signals (with its own UUID)
- Exit configuration (rule-based, continuous signal, or NN-based)
- Backtest parameters (dates, capital, position sizing)
- Execution parameters (slippage, confidence thresholds)

Example strategy file (strategies/my_strategy.json):
{
    "id": "550e8400-e29b-41d4-a716-446655440000",  # UUID for strategy
    "name": "my_strategy",
    "description": "5-class entry with continuous signal exit",
    "created_at": "2024-12-08T16:00:00Z",
    "checkpoints": {
        "lejepa": {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "path": "checkpoints/lejepa_v3/lejepa_best.pt"
        },
        "entry_policy": {
            "id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
            "path": "checkpoints/entry_policy_5class/entry_policy_best.pt"
        },
        "exit_policy": null  # Optional, for NN-based exit
    },
    "exit_mode": "continuous_signal",  # or "rule_based" or "neural_network"
    "exit_config": {
        "stop_loss_pct": 25.0,
        "take_profit_pct": 50.0,
        "eod_exit_minutes": 5.0,
        # For continuous_signal mode:
        "counter_signal_confirmation": 2,
        "plateau_window": 15,
        "plateau_ratio": 0.5,
        "flip_on_counter": true
    },
    "backtest": {
        "initial_capital": 100000,
        "max_position_pct": 0.05,
        "action_cooldown": 1,
        "entry_confidence": 0.4
    },
    "backtests": [
        {
            "id": "bt_20241208_1",
            "date_range": {"start": "2025-01-01", "end": "2025-03-31"},
            "report_path": "reports/backtest_my_strategy_2025Q1.json",
            "metrics": {
                "total_return_pct": 15.5,
                "sharpe_ratio": 1.8,
                "win_rate": 0.62,
                "total_trades": 245
            }
        }
    ]
}
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


STRATEGIES_DIR = Path(__file__).parent.parent.parent / "strategies"


def migrate_strategy_to_uuid_format(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate a strategy from old format to new UUID-based format.

    This handles:
    - Adding UUID id to strategy if missing
    - Converting checkpoint strings to dict with id and path

    Args:
        strategy: Strategy dict (may be old or new format)

    Returns:
        Strategy in new format with UUIDs
    """
    modified = False

    # Add strategy UUID if missing
    if "id" not in strategy:
        strategy["id"] = str(uuid.uuid4())
        modified = True

    # Migrate checkpoints from string to dict format
    checkpoints = strategy.get("checkpoints", {})
    for key in ["lejepa", "entry_policy", "exit_policy"]:
        cp = checkpoints.get(key)
        if cp is not None and isinstance(cp, str):
            checkpoints[key] = {
                "id": str(uuid.uuid4()),
                "path": cp,
            }
            modified = True

    if modified:
        strategy["checkpoints"] = checkpoints

    return strategy


def ensure_strategies_dir() -> None:
    """Ensure strategies directory exists."""
    STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)


def _scan_strategy_files() -> List[Path]:
    """Scan strategies directory for .json files."""
    ensure_strategies_dir()
    return sorted(STRATEGIES_DIR.glob("*.json"))


def list_strategies() -> List[Dict[str, Any]]:
    """List all strategies with summary info."""
    strategies = []

    for strategy_path in _scan_strategy_files():
        try:
            with open(strategy_path) as f:
                strategy = json.load(f)
            name = strategy_path.stem
            strategies.append({
                "id": strategy.get("id"),
                "name": name,
                "description": strategy.get("description", ""),
                "created_at": strategy.get("created_at"),
                "exit_mode": strategy.get("exit_mode"),
                "num_backtests": len(strategy.get("backtests", [])),
                "checkpoints": strategy.get("checkpoints", {}),
            })
        except (json.JSONDecodeError, IOError):
            continue

    return strategies


def get_strategy_by_id(strategy_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a strategy by its UUID.

    Args:
        strategy_id: Strategy UUID

    Returns:
        Strategy dict or None if not found
    """
    for strategy_path in _scan_strategy_files():
        try:
            with open(strategy_path) as f:
                strategy = json.load(f)
            if strategy.get("id") == strategy_id:
                return strategy
        except (json.JSONDecodeError, IOError):
            continue
    return None


def get_strategy(name: str, migrate: bool = True) -> Optional[Dict[str, Any]]:
    """
    Load a strategy by name.

    Args:
        name: Strategy name (without .json extension)
        migrate: If True, migrate old format to new UUID format and save

    Returns:
        Strategy dict or None if not found
    """
    strategy_path = STRATEGIES_DIR / f"{name}.json"
    if not strategy_path.exists():
        return None

    with open(strategy_path) as f:
        strategy = json.load(f)

    # Auto-migrate to UUID format if needed
    if migrate and ("id" not in strategy or _needs_checkpoint_migration(strategy)):
        strategy = migrate_strategy_to_uuid_format(strategy)
        with open(strategy_path, "w") as f:
            json.dump(strategy, f, indent=2)

    return strategy


def _needs_checkpoint_migration(strategy: Dict[str, Any]) -> bool:
    """Check if any checkpoint is still in old string format."""
    checkpoints = strategy.get("checkpoints", {})
    for key in ["lejepa", "entry_policy", "exit_policy"]:
        cp = checkpoints.get(key)
        if cp is not None and isinstance(cp, str):
            return True
    return False


def _make_checkpoint_entry(path: Optional[str]) -> Optional[Dict[str, str]]:
    """Create a checkpoint entry with UUID and path."""
    if path is None:
        return None
    return {
        "id": str(uuid.uuid4()),
        "path": path,
    }


def create_strategy(
    name: str,
    description: str,
    lejepa_checkpoint: str,
    entry_policy_checkpoint: str,
    exit_mode: str = "continuous_signal",
    exit_policy_checkpoint: Optional[str] = None,
    exit_config: Optional[Dict[str, Any]] = None,
    backtest_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a new strategy.

    Args:
        name: Strategy name (used as filename)
        description: Human-readable description
        lejepa_checkpoint: Path to LeJEPA checkpoint
        entry_policy_checkpoint: Path to entry policy checkpoint
        exit_mode: One of "continuous_signal", "rule_based", "neural_network"
        exit_policy_checkpoint: Path to exit policy checkpoint (for NN mode)
        exit_config: Exit policy configuration
        backtest_config: Backtest configuration

    Returns:
        Created strategy dict
    """
    ensure_strategies_dir()

    # Default exit config
    default_exit_config = {
        "stop_loss_pct": 25.0,
        "take_profit_pct": 50.0,
        "eod_exit_minutes": 5.0,
    }
    if exit_mode == "continuous_signal":
        default_exit_config.update({
            "counter_signal_confirmation": 2,
            "plateau_window": 15,
            "plateau_ratio": 0.5,
            "flip_on_counter": True,
        })
    elif exit_mode == "rule_based":
        default_exit_config.update({
            "risk_reward_ratio": 2.0,
            "time_stop_hours": None,
        })

    # Default backtest config
    default_backtest_config = {
        "initial_capital": 100000,
        "max_position_pct": 0.05,
        "action_cooldown": 1,
        "entry_confidence": 0.4,
    }

    strategy = {
        "id": str(uuid.uuid4()),
        "name": name,
        "description": description,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "checkpoints": {
            "lejepa": _make_checkpoint_entry(lejepa_checkpoint),
            "entry_policy": _make_checkpoint_entry(entry_policy_checkpoint),
            "exit_policy": _make_checkpoint_entry(exit_policy_checkpoint),
        },
        "exit_mode": exit_mode,
        "exit_config": {**default_exit_config, **(exit_config or {})},
        "backtest": {**default_backtest_config, **(backtest_config or {})},
        "backtests": [],
    }

    # Save strategy file
    strategy_path = STRATEGIES_DIR / f"{name}.json"
    with open(strategy_path, "w") as f:
        json.dump(strategy, f, indent=2)

    return strategy


def update_strategy(name: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Update an existing strategy.

    Args:
        name: Strategy name
        updates: Dict of updates to apply

    Returns:
        Updated strategy or None if not found
    """
    strategy = get_strategy(name)
    if strategy is None:
        return None

    # Apply updates (shallow merge for top-level keys)
    for key, value in updates.items():
        if isinstance(value, dict) and key in strategy and isinstance(strategy[key], dict):
            strategy[key].update(value)
        else:
            strategy[key] = value

    strategy["updated_at"] = datetime.utcnow().isoformat() + "Z"

    # Save
    strategy_path = STRATEGIES_DIR / f"{name}.json"
    with open(strategy_path, "w") as f:
        json.dump(strategy, f, indent=2)

    return strategy


def add_backtest_result(
    strategy_name: str,
    start_date: str,
    end_date: str,
    report_path: str,
    metrics: Dict[str, float],
) -> Optional[Dict[str, Any]]:
    """
    Add a backtest result to a strategy.

    Args:
        strategy_name: Strategy name
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        report_path: Path to generated HTML report
        metrics: Backtest metrics dict

    Returns:
        Updated strategy or None if not found
    """
    strategy = get_strategy(strategy_name)
    if strategy is None:
        return None

    # Generate backtest ID
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backtest_id = f"bt_{timestamp}"

    backtest_result = {
        "id": backtest_id,
        "run_at": datetime.utcnow().isoformat() + "Z",
        "date_range": {
            "start": start_date,
            "end": end_date,
        },
        "report_path": report_path,
        "metrics": metrics,
    }

    strategy["backtests"].append(backtest_result)

    # Save
    strategy_path = STRATEGIES_DIR / f"{strategy_name}.json"
    with open(strategy_path, "w") as f:
        json.dump(strategy, f, indent=2)

    return strategy


def delete_strategy(name: str) -> bool:
    """
    Delete a strategy.

    Args:
        name: Strategy name

    Returns:
        True if deleted, False if not found
    """
    strategy_path = STRATEGIES_DIR / f"{name}.json"
    if not strategy_path.exists():
        return False

    strategy_path.unlink()
    return True


def get_checkpoint_path(checkpoint: Optional[Dict[str, str] | str]) -> Optional[str]:
    """
    Extract checkpoint path from either old format (string) or new format (dict).

    Args:
        checkpoint: Either a string path or dict with {"id": uuid, "path": path}

    Returns:
        Checkpoint path or None
    """
    if checkpoint is None:
        return None
    if isinstance(checkpoint, str):
        return checkpoint
    if isinstance(checkpoint, dict):
        return checkpoint.get("path")
    return None


def get_checkpoint_id(checkpoint: Optional[Dict[str, str] | str]) -> Optional[str]:
    """
    Extract checkpoint ID from checkpoint entry.

    Args:
        checkpoint: Either a string path or dict with {"id": uuid, "path": path}

    Returns:
        Checkpoint UUID or None
    """
    if checkpoint is None:
        return None
    if isinstance(checkpoint, dict):
        return checkpoint.get("id")
    return None


def strategy_to_args(strategy: Dict[str, Any]) -> List[str]:
    """
    Convert a strategy to command-line arguments for the backtest runner.

    Args:
        strategy: Strategy dict

    Returns:
        List of command-line arguments
    """
    args = []

    # Checkpoints - handle both old format (string) and new format (dict)
    checkpoints = strategy.get("checkpoints", {})
    lejepa_path = get_checkpoint_path(checkpoints.get("lejepa"))
    entry_path = get_checkpoint_path(checkpoints.get("entry_policy"))
    exit_path = get_checkpoint_path(checkpoints.get("exit_policy"))

    if lejepa_path:
        args.extend(["--lejepa", lejepa_path])
    if entry_path:
        args.extend(["--entry_policy", entry_path])
    if exit_path:
        args.extend(["--exit_policy", exit_path])

    # Exit mode
    exit_mode = strategy.get("exit_mode", "continuous_signal")
    exit_config = strategy.get("exit_config", {})

    if exit_mode == "continuous_signal":
        args.append("--continuous_signal")
        if exit_config.get("counter_signal_confirmation"):
            args.extend(["--counter_signal_confirmation", str(exit_config["counter_signal_confirmation"])])
        if exit_config.get("plateau_window"):
            args.extend(["--plateau_window", str(exit_config["plateau_window"])])
        if exit_config.get("plateau_ratio"):
            args.extend(["--plateau_ratio", str(exit_config["plateau_ratio"])])
        if exit_config.get("flip_on_counter") is False:
            args.append("--no_flip_on_counter")
    elif exit_mode == "rule_based":
        args.append("--rule_based_exit")
        if exit_config.get("risk_reward_ratio"):
            args.extend(["--risk_reward_ratio", str(exit_config["risk_reward_ratio"])])
        if exit_config.get("time_stop_hours"):
            args.extend(["--time_stop_hours", str(exit_config["time_stop_hours"])])

    # Common exit config
    if exit_config.get("stop_loss_pct"):
        args.extend(["--stop_loss_pct", str(exit_config["stop_loss_pct"])])
    if exit_config.get("take_profit_pct"):
        args.extend(["--take_profit_pct", str(exit_config["take_profit_pct"])])
    if exit_config.get("eod_exit_minutes"):
        args.extend(["--eod_exit_minutes", str(exit_config["eod_exit_minutes"])])

    # Backtest config
    bt_config = strategy.get("backtest", {})
    if bt_config.get("initial_capital"):
        args.extend(["--initial_capital", str(bt_config["initial_capital"])])
    if bt_config.get("max_position_pct"):
        args.extend(["--max_position_pct", str(bt_config["max_position_pct"])])
    if bt_config.get("action_cooldown"):
        args.extend(["--action_cooldown", str(bt_config["action_cooldown"])])
    if bt_config.get("entry_confidence"):
        args.extend(["--entry_confidence", str(bt_config["entry_confidence"])])

    return args
