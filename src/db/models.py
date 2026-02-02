"""
SQLAlchemy models for SpyMaster database.

These models are compatible with both SQLite and PostgreSQL.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid

from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    ForeignKey,
    JSON,
    Boolean,
    create_engine,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


class Checkpoint(Base):
    """
    Model checkpoint configuration and metadata.

    Data files (.pt models) are stored in data/checkpoints/{name}/
    """
    __tablename__ = "checkpoints"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), unique=True, nullable=False, index=True)
    model_type = Column(String(50), nullable=False, default="lejepa")
    status = Column(String(50), nullable=False, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Hyperparameters (stored as JSON for flexibility)
    hyperparameters = Column(JSON, default=dict)

    # Data configuration
    underlying = Column(String(10), default="SPY")
    start_date = Column(String(10), nullable=True)  # YYYY-MM-DD
    end_date = Column(String(10), nullable=True)

    # Training metrics (updated after training)
    metrics = Column(JSON, default=dict)

    # Path to data directory (relative to project root)
    data_path = Column(String(500), nullable=True)

    # Relationships
    strategies = relationship("Strategy", back_populates="checkpoint")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "model_type": self.model_type,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "hyperparameters": self.hyperparameters or {},
            "underlying": self.underlying,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "metrics": self.metrics or {},
            "data_path": self.data_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        return cls(
            id=data.get("id", generate_uuid()),
            name=data["name"],
            model_type=data.get("model_type", "lejepa"),
            status=data.get("status", "pending"),
            hyperparameters=data.get("hyperparameters", {}),
            underlying=data.get("underlying", "SPY"),
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            metrics=data.get("metrics", {}),
            data_path=data.get("data_path"),
        )


class Strategy(Base):
    """
    Trading strategy configuration.

    Links a checkpoint to strategy parameters.
    """
    __tablename__ = "strategies"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), unique=True, nullable=False, index=True)
    strategy_type = Column(String(50), nullable=False, default="entry_exit")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Link to checkpoint
    checkpoint_id = Column(String(36), ForeignKey("checkpoints.id"), nullable=True)
    checkpoint = relationship("Checkpoint", back_populates="strategies")

    # Strategy parameters (stored as JSON)
    parameters = Column(JSON, default=dict)

    # Active/inactive flag
    is_active = Column(Boolean, default=True)

    # Relationships
    backtests = relationship("Backtest", back_populates="strategy")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "strategy_type": self.strategy_type,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_name": self.checkpoint.name if self.checkpoint else None,
            "parameters": self.parameters or {},
            "is_active": self.is_active,
        }


class Backtest(Base):
    """
    Backtest results and metrics.

    Detailed results (trades, equity curve) stored in data/backtests/{id}/
    """
    __tablename__ = "backtests"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Link to strategy
    strategy_id = Column(String(36), ForeignKey("strategies.id"), nullable=True)
    strategy = relationship("Strategy", back_populates="backtests")

    # Date range
    start_date = Column(String(10), nullable=True)
    end_date = Column(String(10), nullable=True)

    # Initial capital
    initial_capital = Column(Float, default=100000.0)

    # Performance metrics
    total_return_pct = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown_pct = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=True)

    # Additional metrics (stored as JSON)
    metrics = Column(JSON, default=dict)

    # Status
    status = Column(String(50), default="completed")

    # Path to detailed results
    data_path = Column(String(500), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy.name if self.strategy else None,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "total_return_pct": self.total_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "metrics": self.metrics or {},
            "status": self.status,
            "data_path": self.data_path,
        }


# =============================================================================
# Dataset Configuration Models
# =============================================================================


class Dataset(Base):
    """
    Dataset configuration - replaces datasets.json.

    Stores all dataset metadata including:
    - Source datasets (fetched from providers like Polygon, ThetaData)
    - Computed datasets (derived from other datasets)

    Examples:
        # Source dataset
        Dataset(name="stocks-1m", type="source", provider="polygon", ...)

        # Computed dataset
        Dataset(name="training-1m-raw", type="computed", computation="src.data.computations.training_raw:compute", ...)
    """
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), unique=True, nullable=False, index=True)
    type = Column(String(50), nullable=False)  # "source" or "computed"
    description = Column(Text, nullable=True)
    path_pattern = Column(String(500), nullable=False)
    ephemeral = Column(Boolean, default=False)
    version = Column(String(50), nullable=True)
    granularity = Column(String(50), default="daily")  # "daily", "yearly", "full_history"

    # Source dataset fields
    provider = Column(String(50), nullable=True)  # "polygon", "thetadata", etc.
    fetch_config = Column(JSON, default=dict)  # {method, bucket, prefix, expiration, interval, etc.}

    # Computed dataset fields
    computation = Column(String(255), nullable=True)  # Module path like "src.data.computations.gex_flow:compute"

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    fields = relationship("DatasetField", back_populates="dataset", cascade="all, delete-orphan")
    dependencies = relationship(
        "DatasetDependency",
        back_populates="dataset",
        foreign_keys="DatasetDependency.dataset_id",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "path_pattern": self.path_pattern,
            "ephemeral": self.ephemeral,
            "version": self.version,
            "granularity": self.granularity,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

        if self.type == "source":
            result["provider"] = self.provider
            result["fetch_config"] = self.fetch_config or {}
        else:
            result["computation"] = self.computation

        # Include fields
        result["field_map"] = {
            f.name: {
                "type": f.type,
                "description": f.description,
                "source": f.source,
                "unit": f.unit,
            }
            for f in self.fields
        }

        # Include dependencies
        if self.dependencies:
            result["dependencies"] = {
                d.dependency_name: {
                    "relation": d.relation,
                    "days": d.days,
                    "include_current": d.include_current,
                    "required": d.required,
                }
                for d in self.dependencies
            }

        return result

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "Dataset":
        """Create from dictionary (for migration from datasets.json)."""
        return cls(
            name=name,
            type=data.get("type", "source"),
            description=data.get("description"),
            path_pattern=data.get("path_pattern", ""),
            ephemeral=data.get("ephemeral", False),
            version=data.get("version"),
            granularity=data.get("granularity", "daily"),
            provider=data.get("provider"),
            fetch_config=data.get("fetch_config", {}),
            computation=data.get("computation"),
        )


class DatasetField(Base):
    """
    Field definition for a dataset schema.

    Maps internal field names to source column names with type information.

    Examples:
        DatasetField(name="timestamp", type="datetime", source="window_start", unit="ns")
        DatasetField(name="close", type="float64", description="Closing price")
    """
    __tablename__ = "dataset_fields"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)  # "datetime", "float64", "int64", "string"
    description = Column(Text, nullable=True)
    source = Column(String(255), nullable=True)  # Source column name if different from name
    unit = Column(String(50), nullable=True)  # Unit for datetime fields (e.g., "ns")

    # Ensure unique field names per dataset
    __table_args__ = (
        UniqueConstraint("dataset_id", "name", name="uq_dataset_field_name"),
    )

    # Relationship
    dataset = relationship("Dataset", back_populates="fields")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "type": self.type,
            "description": self.description,
        }
        if self.source:
            result["source"] = self.source
        if self.unit:
            result["unit"] = self.unit
        return result

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any], dataset_id: str) -> "DatasetField":
        """Create from dictionary."""
        return cls(
            dataset_id=dataset_id,
            name=name,
            type=data.get("type", "string"),
            description=data.get("description"),
            source=data.get("source"),
            unit=data.get("unit"),
        )


class DatasetDependency(Base):
    """
    Dependency relationship between datasets.

    Defines how a computed dataset depends on other datasets.

    Relation types:
    - same_day: 1:1 mapping, need data from same date
    - lookback: Need N days prior (for moving averages)
    - all_history: Need all available historical data

    Examples:
        # Same-day dependency
        DatasetDependency(dependency_name="stocks-1m", relation="same_day")

        # Lookback dependency (need 50 trading days prior for SMA calculation)
        DatasetDependency(dependency_name="stocks-1d", relation="lookback", days=50, include_current=True)
    """
    __tablename__ = "dataset_dependencies"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    dependency_name = Column(String(255), nullable=False)  # Name of the dependency dataset
    relation = Column(String(50), default="same_day")  # "same_day", "lookback", "all_history"
    days = Column(Integer, default=0)  # For lookback: number of days
    include_current = Column(Boolean, default=True)  # For lookback: include current day?
    required = Column(Boolean, default=True)  # Is this dependency required?

    # Ensure unique dependency per dataset
    __table_args__ = (
        UniqueConstraint("dataset_id", "dependency_name", name="uq_dataset_dependency"),
    )

    # Relationship
    dataset = relationship("Dataset", back_populates="dependencies", foreign_keys=[dataset_id])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "relation": self.relation,
            "days": self.days,
            "include_current": self.include_current,
            "required": self.required,
        }

    @classmethod
    def from_dict(
        cls,
        dependency_name: str,
        data: Dict[str, Any],
        dataset_id: str,
    ) -> "DatasetDependency":
        """Create from dictionary."""
        # Handle both old format (list of strings) and new format (dict)
        if isinstance(data, str):
            # Old format: just the relation type
            return cls(
                dataset_id=dataset_id,
                dependency_name=dependency_name,
                relation=data,
            )
        return cls(
            dataset_id=dataset_id,
            dependency_name=dependency_name,
            relation=data.get("relation", "same_day"),
            days=data.get("days", 0),
            include_current=data.get("include_current", True),
            required=data.get("required", True),
        )


class DatasetDefaults(Base):
    """
    Default configuration values for datasets.

    Stores global defaults like underlying symbol and format.
    Only one row should exist in this table.
    """
    __tablename__ = "dataset_defaults"

    id = Column(Integer, primary_key=True, default=1)
    underlying = Column(String(10), default="SPY")
    format = Column(String(50), default="parquet")
    granularity = Column(String(50), default="daily")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "underlying": self.underlying,
            "format": self.format,
            "granularity": self.granularity,
        }


class LivePositionState(Base):
    """
    Live trading position state for persistence.

    Replaces JSON file storage for position state. Stores current active
    position and trading session state for recovery after restarts.

    Examples:
        # Active call position
        LivePositionState(
            session_id="paper_20250127",
            has_active_position=True,
            position_type="call",
            option_symbol="SPY250127C00600000",
            ...
        )
    """
    __tablename__ = "live_position_state"

    # Single row per trading session
    session_id = Column(String(50), primary_key=True, default="default")

    # Active position data (nullable when no position)
    has_active_position = Column(Boolean, default=False)
    position_id = Column(String(100), nullable=True)
    position_type = Column(String(10), nullable=True)  # "call" or "put"
    entry_time = Column(DateTime, nullable=True)
    entry_option_price = Column(Float, nullable=True)
    option_symbol = Column(String(50), nullable=True)
    strike = Column(Float, nullable=True)
    num_contracts = Column(Integer, nullable=True)
    entry_underlying_price = Column(Float, nullable=True)
    position_value = Column(Float, nullable=True)
    dominant_model = Column(String(50), nullable=True)
    max_hold_minutes = Column(Integer, nullable=True)
    barrier_start_time = Column(DateTime, nullable=True)
    confluence_count = Column(Integer, default=2)
    peak_pnl_pct = Column(Float, default=0.0)
    breakeven_activated = Column(Boolean, default=False)
    is_runner = Column(Boolean, default=False)
    runner_start_time = Column(DateTime, nullable=True)
    runner_max_hold_minutes = Column(Integer, default=15)
    runner_peak_pnl_pct = Column(Float, default=0.0)
    runner_entry_pnl_pct = Column(Float, default=0.0)
    is_breach = Column('is_vwap_breach', Boolean, default=False)
    is_reversal = Column(Boolean, default=False)
    is_bounce = Column(Boolean, default=False)
    renewals = Column(Integer, default=0)

    # Session state
    daily_trades = Column(Integer, default=0)
    daily_pnl = Column(Float, default=0.0)
    capital = Column(Float, default=25000.0)
    last_trade_date = Column(String(10), nullable=True)

    # Recent completed trades (JSON array of trade dicts)
    completed_trades_json = Column(JSON, default=list)

    # Timestamps
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy access."""
        return {
            "session_id": self.session_id,
            "has_active_position": self.has_active_position,
            "position_id": self.position_id,
            "position_type": self.position_type,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_option_price": self.entry_option_price,
            "option_symbol": self.option_symbol,
            "strike": self.strike,
            "num_contracts": self.num_contracts,
            "entry_underlying_price": self.entry_underlying_price,
            "position_value": self.position_value,
            "dominant_model": self.dominant_model,
            "max_hold_minutes": self.max_hold_minutes,
            "barrier_start_time": self.barrier_start_time.isoformat() if self.barrier_start_time else None,
            "confluence_count": self.confluence_count,
            "peak_pnl_pct": self.peak_pnl_pct,
            "breakeven_activated": self.breakeven_activated,
            "is_runner": self.is_runner,
            "runner_start_time": self.runner_start_time.isoformat() if self.runner_start_time else None,
            "runner_max_hold_minutes": self.runner_max_hold_minutes,
            "runner_peak_pnl_pct": self.runner_peak_pnl_pct,
            "runner_entry_pnl_pct": self.runner_entry_pnl_pct,
            "is_breach": self.is_breach,
            "is_reversal": self.is_reversal,
            "is_bounce": self.is_bounce,
            "renewals": self.renewals,
            "daily_trades": self.daily_trades,
            "daily_pnl": self.daily_pnl,
            "capital": self.capital,
            "last_trade_date": self.last_trade_date,
            "completed_trades": self.completed_trades_json or [],
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class MarketCalendar(Base):
    """
    Market calendar tracking for trading days.

    Tracks market schedule for each day using ThetaData calendar API.
    Schedule types:
    - open: Regular trading day (9:30 AM - 4:00 PM ET)
    - early_close: Early close day (9:30 AM - 1:00 PM ET)
    - full_close: Market closed (holiday)
    - weekend: Saturday/Sunday

    All times are stored as strings in HH:MM:SS format (ET timezone).

    Examples:
        # Regular trading day
        MarketCalendar(date="2024-12-16", schedule_type="open",
                       market_open="09:30:00", market_close="16:00:00")

        # Early close (day after Thanksgiving)
        MarketCalendar(date="2024-11-29", schedule_type="early_close",
                       market_open="09:30:00", market_close="13:00:00")

        # Holiday
        MarketCalendar(date="2024-12-25", schedule_type="full_close",
                       market_open=None, market_close=None)
    """
    __tablename__ = "market_calendar"

    # Use date as primary key (YYYY-MM-DD format)
    date = Column(String(10), primary_key=True)

    # Schedule type from ThetaData: "open", "early_close", "full_close", "weekend"
    schedule_type = Column(String(20), nullable=False, index=True)

    # Market hours in ET timezone (HH:MM:SS format)
    # Null for full_close and weekend
    market_open = Column(String(8), nullable=True)
    market_close = Column(String(8), nullable=True)

    # Timestamp when this record was fetched/updated
    fetched_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "date": self.date,
            "schedule_type": self.schedule_type,
            "market_open": self.market_open,
            "market_close": self.market_close,
            "fetched_at": self.fetched_at.isoformat() if self.fetched_at else None,
        }

    @property
    def is_trading_day(self) -> bool:
        """Check if this is a trading day (open or early_close)."""
        return self.schedule_type in ("open", "early_close")

    @property
    def trading_minutes(self) -> int:
        """Get number of trading minutes for this day."""
        if not self.is_trading_day:
            return 0
        # Regular day: 9:30 AM - 4:00 PM = 390 minutes
        # Early close: 9:30 AM - 1:00 PM = 210 minutes
        if self.schedule_type == "early_close":
            return 210
        return 390

    @classmethod
    def from_api_response(cls, date_str: str, api_response: Dict[str, Any]) -> "MarketCalendar":
        """
        Create from ThetaData calendar API response.

        API response format:
        {
            "type": "open",
            "open": "09:30:00",
            "close": "16:00:00"
        }
        """
        return cls(
            date=date_str,
            schedule_type=api_response.get("type", "full_close"),
            market_open=api_response.get("open"),
            market_close=api_response.get("close"),
        )
