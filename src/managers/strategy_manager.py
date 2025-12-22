"""Strategy manager - handles CRUD operations for trading strategies."""

import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


def _generate_id() -> str:
    """Generate a new UUID for a record."""
    return str(uuid.uuid4())


@dataclass
class StrategyConfig:
    """Configuration for a strategy."""
    name: str
    id: str = field(default_factory=_generate_id)
    strategy_type: str = "entry_exit"
    checkpoint: str = ""
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.strategy_type,
            "checkpoint": self.checkpoint,
            "created": self.created,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyConfig":
        return cls(
            id=data.get("id", _generate_id()),
            name=data.get("name", "unknown"),
            strategy_type=data.get("type", "entry_exit"),
            checkpoint=data.get("checkpoint", ""),
            created=data.get("created", ""),
            parameters=data.get("parameters", {}),
        )


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    backtest_id: str
    strategy_name: str
    start_date: str
    end_date: str
    sharpe_ratio: float
    total_return_pct: float
    win_rate: float
    max_drawdown: float
    total_trades: int
    created: float
    raw_data: Dict[str, Any]


@dataclass
class StrategyInfo:
    """Information about a strategy."""
    name: str
    path: Path
    config: Optional[StrategyConfig]
    modified: float
    backtest_count: int
    files: List[Path]


class StrategyManager:
    """Manager for trading strategies."""

    def __init__(
        self,
        strategies_dir: Path,
        backtests_dir: Path,
        checkpoints_dir: Path,
        project_root: Path,
    ):
        self.strategies_dir = strategies_dir
        self.backtests_dir = backtests_dir
        self.checkpoints_dir = checkpoints_dir
        self.project_root = project_root

        self.strategies_dir.mkdir(exist_ok=True)
        self.backtests_dir.mkdir(exist_ok=True)

    def _load_config(self, strategy_dir: Path) -> Optional[StrategyConfig]:
        """Load strategy configuration.

        If the config doesn't have an ID, one will be generated and saved.
        """
        config_file = strategy_dir / "strategy.json"
        if config_file.exists():
            with open(config_file) as f:
                data = json.load(f)

            # Check if ID needs to be generated (migration for existing records)
            needs_save = "id" not in data
            config = StrategyConfig.from_dict(data)

            # Persist the generated ID
            if needs_save:
                self._save_config(strategy_dir, config)

            return config
        return None

    def _save_config(self, strategy_dir: Path, config: StrategyConfig) -> None:
        """Save strategy configuration."""
        config_file = strategy_dir / "strategy.json"
        with open(config_file, "w") as f:
            json.dump(config.to_dict(), f, indent=2)

    def _get_backtest_count(self, name: str) -> int:
        """Get the number of backtests for a strategy."""
        backtest_dir = self.backtests_dir / name
        if backtest_dir.exists():
            return len(list(backtest_dir.glob("*.json")))
        return 0

    def list_all(self) -> List[StrategyInfo]:
        """List all strategies."""
        strategies = []

        if not self.strategies_dir.exists():
            return strategies

        for strategy_path in sorted(self.strategies_dir.iterdir()):
            if strategy_path.is_dir():
                info = self.get(strategy_path.name)
                if info:
                    strategies.append(info)

        return strategies

    def get(self, name: str) -> Optional[StrategyInfo]:
        """Get strategy information."""
        strategy_dir = self.strategies_dir / name

        if not strategy_dir.exists():
            return None

        config = self._load_config(strategy_dir)

        return StrategyInfo(
            name=name,
            path=strategy_dir,
            config=config,
            modified=strategy_dir.stat().st_mtime,
            backtest_count=self._get_backtest_count(name),
            files=[f for f in strategy_dir.iterdir() if f.is_file()],
        )

    def exists(self, name: str) -> bool:
        """Check if strategy exists."""
        return (self.strategies_dir / name).exists()

    def checkpoint_exists(self, checkpoint: str) -> bool:
        """Check if checkpoint exists."""
        return (self.checkpoints_dir / checkpoint).exists()

    def create(
        self,
        name: str,
        checkpoint: str,
        strategy_type: str = "entry_exit",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> StrategyConfig:
        """Create a new strategy."""
        strategy_dir = self.strategies_dir / name
        strategy_dir.mkdir(parents=True)

        config = StrategyConfig(
            name=name,
            strategy_type=strategy_type,
            checkpoint=checkpoint,
            parameters=parameters or {},
        )

        self._save_config(strategy_dir, config)
        return config

    def remove(self, name: str, remove_backtests: bool = True) -> bool:
        """Remove a strategy."""
        strategy_dir = self.strategies_dir / name

        if not strategy_dir.exists():
            return False

        shutil.rmtree(strategy_dir)

        # Also remove backtests if requested
        if remove_backtests:
            backtest_dir = self.backtests_dir / name
            if backtest_dir.exists():
                shutil.rmtree(backtest_dir)

        return True

    def list_backtests(self, name: str) -> List[BacktestResult]:
        """List all backtests for a strategy."""
        backtests = []
        backtest_dir = self.backtests_dir / name

        if not backtest_dir.exists():
            return backtests

        for backtest_file in sorted(backtest_dir.glob("*.json")):
            result = self.get_backtest(name, backtest_file.stem)
            if result:
                backtests.append(result)

        return backtests

    def get_backtest(self, strategy_name: str, backtest_id: str) -> Optional[BacktestResult]:
        """Get a specific backtest result."""
        backtest_file = self.backtests_dir / strategy_name / f"{backtest_id}.json"

        if not backtest_file.exists():
            return None

        with open(backtest_file) as f:
            data = json.load(f)

        return BacktestResult(
            backtest_id=backtest_id,
            strategy_name=strategy_name,
            start_date=data.get("start_date", ""),
            end_date=data.get("end_date", ""),
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            total_return_pct=data.get("total_return_pct", 0.0),
            win_rate=data.get("win_rate", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            total_trades=data.get("total_trades", 0),
            created=backtest_file.stat().st_mtime,
            raw_data=data,
        )

    def build_backtest_command(
        self,
        name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 100000.0,
    ) -> Optional[List[str]]:
        """Build the backtest command for a strategy."""
        info = self.get(name)
        if not info or not info.config:
            return None

        checkpoint = info.config.checkpoint
        if not checkpoint:
            return None

        cmd = [
            "uv", "run", "python", "-m", "src.backtest.runner",
            "--checkpoint", str(self.checkpoints_dir / checkpoint),
            "--initial-capital", str(initial_capital),
        ]

        if start_date:
            cmd.extend(["--start-date", start_date])
        if end_date:
            cmd.extend(["--end-date", end_date])

        return cmd

    def run_backtest(
        self,
        name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 100000.0,
    ) -> Optional[subprocess.CompletedProcess]:
        """Run a backtest for a strategy."""
        cmd = self.build_backtest_command(name, start_date, end_date, initial_capital)
        if not cmd:
            return None

        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.project_root,
        )

    def save_backtest_result(
        self,
        strategy_name: str,
        result: Dict[str, Any],
        backtest_id: Optional[str] = None,
    ) -> str:
        """Save a backtest result."""
        backtest_dir = self.backtests_dir / strategy_name
        backtest_dir.mkdir(parents=True, exist_ok=True)

        if backtest_id is None:
            backtest_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        backtest_file = backtest_dir / f"{backtest_id}.json"
        with open(backtest_file, "w") as f:
            json.dump(result, f, indent=2)

        return backtest_id

    def update(
        self,
        name: str,
        checkpoint: Optional[str] = None,
        strategy_type: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[StrategyConfig]:
        """Update an existing strategy configuration.

        Only non-None values will be updated.
        """
        strategy_dir = self.strategies_dir / name

        if not strategy_dir.exists():
            return None

        config = self._load_config(strategy_dir)
        if not config:
            return None

        if checkpoint is not None:
            config.checkpoint = checkpoint
        if strategy_type is not None:
            config.strategy_type = strategy_type
        if parameters is not None:
            config.parameters.update(parameters)

        self._save_config(strategy_dir, config)
        return config

    def get_config_path(self, name: str) -> Optional[Path]:
        """Get the path to the strategy.json file for a strategy."""
        strategy_dir = self.strategies_dir / name
        config_file = strategy_dir / "strategy.json"
        if config_file.exists():
            return config_file
        return None
