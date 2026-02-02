"""
Database manager for SpyMaster.

Provides a unified interface for SQLite (development) and PostgreSQL (production).
"""

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session

from src.db.models import (
    Base,
    Checkpoint,
    Strategy,
    Backtest,
    Dataset,
    DatasetField,
    DatasetDependency,
    DatasetDefaults,
    MarketCalendar,
    LivePositionState,
    generate_uuid,
)


# Default database path (SQLite)
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent / "data" / "spymaster.db"


class Database:
    """
    Database manager with support for SQLite and PostgreSQL.

    Usage:
        db = Database()  # Uses SQLite at data/spymaster.db
        db = Database("postgresql://user:pass@localhost/spymaster")  # PostgreSQL

        # Create records
        checkpoint = db.create_checkpoint(name="lejepa-v1", model_type="lejepa")

        # Query records
        checkpoints = db.list_checkpoints()
        checkpoint = db.get_checkpoint("lejepa-v1")

        # Update records
        db.update_checkpoint("lejepa-v1", status="trained", metrics={"loss": 0.1})

        # Delete records
        db.delete_checkpoint("lejepa-v1")
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            database_url: SQLAlchemy database URL. If None, uses SQLite at DEFAULT_DB_PATH.
                         Set SPYMASTER_DATABASE_URL env var to override.
        """
        if database_url is None:
            database_url = os.environ.get("SPYMASTER_DATABASE_URL")

        if database_url is None:
            # Default to SQLite
            DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            database_url = f"sqlite:///{DEFAULT_DB_PATH}"

        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            # SQLite specific: enable foreign keys
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
        )

        # Enable foreign keys for SQLite
        if "sqlite" in database_url:
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(self.engine)

    def drop_tables(self) -> None:
        """Drop all database tables. USE WITH CAUTION."""
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Get a database session context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # =========================================================================
    # Checkpoint Operations
    # =========================================================================

    def create_checkpoint(
        self,
        name: str,
        model_type: str = "lejepa",
        status: str = "pending",
        hyperparameters: Optional[Dict[str, Any]] = None,
        underlying: str = "SPY",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data_path: Optional[str] = None,
    ) -> Checkpoint:
        """Create a new checkpoint."""
        with self.session() as session:
            checkpoint = Checkpoint(
                name=name,
                model_type=model_type,
                status=status,
                hyperparameters=hyperparameters or {},
                underlying=underlying,
                start_date=start_date,
                end_date=end_date,
                data_path=data_path or f"data/checkpoints/{name}",
            )
            session.add(checkpoint)
            session.flush()
            # Detach from session before returning
            session.expunge(checkpoint)
            return checkpoint

    def get_checkpoint(self, name: str) -> Optional[Checkpoint]:
        """Get a checkpoint by name."""
        with self.session() as session:
            checkpoint = session.query(Checkpoint).filter(Checkpoint.name == name).first()
            if checkpoint:
                session.expunge(checkpoint)
            return checkpoint

    def get_checkpoint_by_id(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a checkpoint by ID."""
        with self.session() as session:
            checkpoint = session.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
            if checkpoint:
                session.expunge(checkpoint)
            return checkpoint

    def list_checkpoints(self) -> List[Checkpoint]:
        """List all checkpoints."""
        with self.session() as session:
            checkpoints = session.query(Checkpoint).order_by(Checkpoint.created_at.desc()).all()
            for cp in checkpoints:
                session.expunge(cp)
            return checkpoints

    def update_checkpoint(
        self,
        name: str,
        **kwargs,
    ) -> Optional[Checkpoint]:
        """Update a checkpoint by name."""
        with self.session() as session:
            checkpoint = session.query(Checkpoint).filter(Checkpoint.name == name).first()
            if not checkpoint:
                return None

            # Update allowed fields
            allowed_fields = {
                "model_type", "status", "hyperparameters", "underlying",
                "start_date", "end_date", "metrics", "data_path",
            }
            for key, value in kwargs.items():
                if key in allowed_fields and value is not None:
                    setattr(checkpoint, key, value)

            checkpoint.updated_at = datetime.utcnow()
            session.flush()
            session.expunge(checkpoint)
            return checkpoint

    def delete_checkpoint(self, name: str) -> bool:
        """Delete a checkpoint by name."""
        with self.session() as session:
            checkpoint = session.query(Checkpoint).filter(Checkpoint.name == name).first()
            if checkpoint:
                session.delete(checkpoint)
                return True
            return False

    def checkpoint_exists(self, name: str) -> bool:
        """Check if a checkpoint exists."""
        with self.session() as session:
            return session.query(Checkpoint).filter(Checkpoint.name == name).count() > 0

    # =========================================================================
    # Strategy Operations
    # =========================================================================

    def create_strategy(
        self,
        name: str,
        strategy_type: str = "entry_exit",
        checkpoint_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Strategy:
        """Create a new strategy."""
        with self.session() as session:
            # Look up checkpoint if name provided
            checkpoint_id = None
            if checkpoint_name:
                checkpoint = session.query(Checkpoint).filter(Checkpoint.name == checkpoint_name).first()
                if checkpoint:
                    checkpoint_id = checkpoint.id

            strategy = Strategy(
                name=name,
                strategy_type=strategy_type,
                checkpoint_id=checkpoint_id,
                parameters=parameters or {},
            )
            session.add(strategy)
            session.flush()
            session.expunge(strategy)
            return strategy

    def get_strategy(self, name: str) -> Optional[Strategy]:
        """Get a strategy by name."""
        with self.session() as session:
            strategy = session.query(Strategy).filter(Strategy.name == name).first()
            if strategy:
                # Load checkpoint relationship
                if strategy.checkpoint:
                    session.expunge(strategy.checkpoint)
                session.expunge(strategy)
            return strategy

    def list_strategies(self) -> List[Strategy]:
        """List all strategies."""
        with self.session() as session:
            strategies = session.query(Strategy).order_by(Strategy.created_at.desc()).all()
            for s in strategies:
                if s.checkpoint:
                    session.expunge(s.checkpoint)
                session.expunge(s)
            return strategies

    def update_strategy(
        self,
        name: str,
        **kwargs,
    ) -> Optional[Strategy]:
        """Update a strategy by name."""
        with self.session() as session:
            strategy = session.query(Strategy).filter(Strategy.name == name).first()
            if not strategy:
                return None

            # Handle checkpoint_name -> checkpoint_id conversion
            if "checkpoint_name" in kwargs:
                checkpoint_name = kwargs.pop("checkpoint_name")
                if checkpoint_name:
                    checkpoint = session.query(Checkpoint).filter(Checkpoint.name == checkpoint_name).first()
                    if checkpoint:
                        strategy.checkpoint_id = checkpoint.id

            # Update allowed fields
            allowed_fields = {"strategy_type", "parameters", "is_active"}
            for key, value in kwargs.items():
                if key in allowed_fields and value is not None:
                    setattr(strategy, key, value)

            strategy.updated_at = datetime.utcnow()
            session.flush()
            session.expunge(strategy)
            return strategy

    def delete_strategy(self, name: str) -> bool:
        """Delete a strategy by name."""
        with self.session() as session:
            strategy = session.query(Strategy).filter(Strategy.name == name).first()
            if strategy:
                session.delete(strategy)
                return True
            return False

    def strategy_exists(self, name: str) -> bool:
        """Check if a strategy exists."""
        with self.session() as session:
            return session.query(Strategy).filter(Strategy.name == name).count() > 0

    # =========================================================================
    # Backtest Operations
    # =========================================================================

    def create_backtest(
        self,
        strategy_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        initial_capital: float = 100000.0,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Backtest:
        """Create a new backtest record."""
        with self.session() as session:
            # Look up strategy if name provided
            strategy_id = None
            if strategy_name:
                strategy = session.query(Strategy).filter(Strategy.name == strategy_name).first()
                if strategy:
                    strategy_id = strategy.id

            backtest = Backtest(
                strategy_id=strategy_id,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                metrics=metrics or {},
            )
            session.add(backtest)
            session.flush()

            # Set data path using generated ID
            backtest.data_path = f"data/backtests/{backtest.id}"
            session.flush()
            session.expunge(backtest)
            return backtest

    def get_backtest(self, backtest_id: str) -> Optional[Backtest]:
        """Get a backtest by ID."""
        with self.session() as session:
            backtest = session.query(Backtest).filter(Backtest.id == backtest_id).first()
            if backtest:
                if backtest.strategy:
                    session.expunge(backtest.strategy)
                session.expunge(backtest)
            return backtest

    def list_backtests(self, strategy_name: Optional[str] = None) -> List[Backtest]:
        """List backtests, optionally filtered by strategy."""
        with self.session() as session:
            query = session.query(Backtest).order_by(Backtest.created_at.desc())

            if strategy_name:
                strategy = session.query(Strategy).filter(Strategy.name == strategy_name).first()
                if strategy:
                    query = query.filter(Backtest.strategy_id == strategy.id)

            backtests = query.all()
            for bt in backtests:
                if bt.strategy:
                    session.expunge(bt.strategy)
                session.expunge(bt)
            return backtests

    def update_backtest(
        self,
        backtest_id: str,
        **kwargs,
    ) -> Optional[Backtest]:
        """Update a backtest by ID."""
        with self.session() as session:
            backtest = session.query(Backtest).filter(Backtest.id == backtest_id).first()
            if not backtest:
                return None

            # Update allowed fields
            allowed_fields = {
                "total_return_pct", "sharpe_ratio", "max_drawdown_pct",
                "win_rate", "profit_factor", "total_trades", "metrics", "status",
            }
            for key, value in kwargs.items():
                if key in allowed_fields and value is not None:
                    setattr(backtest, key, value)

            session.flush()
            session.expunge(backtest)
            return backtest

    def delete_backtest(self, backtest_id: str) -> bool:
        """Delete a backtest by ID."""
        with self.session() as session:
            backtest = session.query(Backtest).filter(Backtest.id == backtest_id).first()
            if backtest:
                session.delete(backtest)
                return True
            return False

    # =========================================================================
    # Dataset Operations
    # =========================================================================

    def create_dataset(
        self,
        name: str,
        type: str,
        path_pattern: str,
        description: Optional[str] = None,
        ephemeral: bool = False,
        version: Optional[str] = None,
        granularity: str = "daily",
        provider: Optional[str] = None,
        fetch_config: Optional[Dict[str, Any]] = None,
        computation: Optional[str] = None,
        field_map: Optional[Dict[str, Dict[str, Any]]] = None,
        dependencies: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dataset:
        """Create a new dataset configuration."""
        with self.session() as session:
            dataset = Dataset(
                name=name,
                type=type,
                path_pattern=path_pattern,
                description=description,
                ephemeral=ephemeral,
                version=version,
                granularity=granularity,
                provider=provider,
                fetch_config=fetch_config or {},
                computation=computation,
            )
            session.add(dataset)
            session.flush()  # Get the ID

            # Add fields
            if field_map:
                for field_name, field_data in field_map.items():
                    field = DatasetField.from_dict(field_name, field_data, dataset.id)
                    session.add(field)

            # Add dependencies
            if dependencies:
                for dep_name, dep_data in dependencies.items():
                    dep = DatasetDependency.from_dict(dep_name, dep_data, dataset.id)
                    session.add(dep)

            session.flush()
            session.expunge(dataset)
            return dataset

    def get_dataset(self, name: str) -> Optional[Dataset]:
        """Get a dataset by name with all relationships loaded."""
        with self.session() as session:
            dataset = (
                session.query(Dataset)
                .filter(Dataset.name == name)
                .first()
            )
            if dataset:
                # Force load relationships
                _ = dataset.fields
                _ = dataset.dependencies
                for f in dataset.fields:
                    session.expunge(f)
                for d in dataset.dependencies:
                    session.expunge(d)
                session.expunge(dataset)
            return dataset

    def list_datasets(self, type: Optional[str] = None) -> List[Dataset]:
        """List all datasets, optionally filtered by type."""
        with self.session() as session:
            query = session.query(Dataset).order_by(Dataset.name)
            if type:
                query = query.filter(Dataset.type == type)

            datasets = query.all()
            for ds in datasets:
                # Force load relationships
                _ = ds.fields
                _ = ds.dependencies
                for f in ds.fields:
                    session.expunge(f)
                for d in ds.dependencies:
                    session.expunge(d)
                session.expunge(ds)
            return datasets

    def update_dataset(
        self,
        name: str,
        **kwargs,
    ) -> Optional[Dataset]:
        """Update a dataset by name."""
        with self.session() as session:
            dataset = session.query(Dataset).filter(Dataset.name == name).first()
            if not dataset:
                return None

            # Handle field_map update
            if "field_map" in kwargs:
                field_map = kwargs.pop("field_map")
                # Delete existing fields
                session.query(DatasetField).filter(DatasetField.dataset_id == dataset.id).delete()
                # Add new fields
                if field_map:
                    for field_name, field_data in field_map.items():
                        field = DatasetField.from_dict(field_name, field_data, dataset.id)
                        session.add(field)

            # Handle dependencies update
            if "dependencies" in kwargs:
                dependencies = kwargs.pop("dependencies")
                # Delete existing dependencies
                session.query(DatasetDependency).filter(DatasetDependency.dataset_id == dataset.id).delete()
                # Add new dependencies
                if dependencies:
                    for dep_name, dep_data in dependencies.items():
                        dep = DatasetDependency.from_dict(dep_name, dep_data, dataset.id)
                        session.add(dep)

            # Update allowed scalar fields
            allowed_fields = {
                "type", "description", "path_pattern", "ephemeral", "version",
                "granularity", "provider", "fetch_config", "computation",
            }
            for key, value in kwargs.items():
                if key in allowed_fields and value is not None:
                    setattr(dataset, key, value)

            dataset.updated_at = datetime.utcnow()
            session.flush()

            # Reload relationships
            _ = dataset.fields
            _ = dataset.dependencies
            for f in dataset.fields:
                session.expunge(f)
            for d in dataset.dependencies:
                session.expunge(d)
            session.expunge(dataset)
            return dataset

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset by name (cascades to fields and dependencies)."""
        with self.session() as session:
            dataset = session.query(Dataset).filter(Dataset.name == name).first()
            if dataset:
                session.delete(dataset)
                return True
            return False

    def dataset_exists(self, name: str) -> bool:
        """Check if a dataset exists."""
        with self.session() as session:
            return session.query(Dataset).filter(Dataset.name == name).count() > 0

    def get_dataset_defaults(self) -> DatasetDefaults:
        """Get or create dataset defaults."""
        with self.session() as session:
            defaults = session.query(DatasetDefaults).first()
            if not defaults:
                defaults = DatasetDefaults()
                session.add(defaults)
                session.flush()
            session.expunge(defaults)
            return defaults

    def update_dataset_defaults(
        self,
        underlying: Optional[str] = None,
        format: Optional[str] = None,
        granularity: Optional[str] = None,
    ) -> DatasetDefaults:
        """Update dataset defaults."""
        with self.session() as session:
            defaults = session.query(DatasetDefaults).first()
            if not defaults:
                defaults = DatasetDefaults()
                session.add(defaults)

            if underlying:
                defaults.underlying = underlying
            if format:
                defaults.format = format
            if granularity:
                defaults.granularity = granularity

            session.flush()
            session.expunge(defaults)
            return defaults

    def import_datasets_from_json(self, json_path: Path) -> int:
        """
        Import datasets from a JSON file (for migration).

        Args:
            json_path: Path to datasets.json

        Returns:
            Number of datasets imported.
        """
        import json

        with open(json_path) as f:
            data = json.load(f)

        # Import defaults
        if "defaults" in data:
            self.update_dataset_defaults(**data["defaults"])

        # Import datasets
        count = 0
        for name, ds_data in data.get("datasets", {}).items():
            if self.dataset_exists(name):
                continue

            self.create_dataset(
                name=name,
                type=ds_data.get("type", "source"),
                path_pattern=ds_data.get("path_pattern", ""),
                description=ds_data.get("description"),
                ephemeral=ds_data.get("ephemeral", False),
                version=ds_data.get("version"),
                granularity=ds_data.get("granularity", "daily"),
                provider=ds_data.get("provider"),
                fetch_config=ds_data.get("fetch_config"),
                computation=ds_data.get("computation"),
                field_map=ds_data.get("field_map"),
                dependencies=ds_data.get("dependencies"),
            )
            count += 1

        return count

    # =========================================================================
    # Market Calendar Operations
    # =========================================================================

    def upsert_calendar_day(
        self,
        date_str: str,
        schedule_type: str,
        market_open: Optional[str] = None,
        market_close: Optional[str] = None,
    ) -> MarketCalendar:
        """
        Insert or update a calendar day.

        Args:
            date_str: Date in YYYY-MM-DD format
            schedule_type: "open", "early_close", "full_close", or "weekend"
            market_open: Opening time in HH:MM:SS format (ET), or None
            market_close: Closing time in HH:MM:SS format (ET), or None

        Returns:
            The created/updated MarketCalendar record.
        """
        with self.session() as session:
            existing = session.query(MarketCalendar).filter(
                MarketCalendar.date == date_str
            ).first()

            if existing:
                existing.schedule_type = schedule_type
                existing.market_open = market_open
                existing.market_close = market_close
                existing.fetched_at = datetime.utcnow()
                session.flush()
                session.expunge(existing)
                return existing
            else:
                calendar = MarketCalendar(
                    date=date_str,
                    schedule_type=schedule_type,
                    market_open=market_open,
                    market_close=market_close,
                )
                session.add(calendar)
                session.flush()
                session.expunge(calendar)
                return calendar

    def upsert_calendar_from_api(
        self,
        date_str: str,
        api_response: Dict[str, Any],
    ) -> MarketCalendar:
        """
        Insert or update a calendar day from ThetaData API response.

        Args:
            date_str: Date in YYYY-MM-DD format
            api_response: ThetaData calendar API response with "type", "open", "close"

        Returns:
            The created/updated MarketCalendar record.
        """
        return self.upsert_calendar_day(
            date_str=date_str,
            schedule_type=api_response.get("type", "full_close"),
            market_open=api_response.get("open"),
            market_close=api_response.get("close"),
        )

    def get_calendar_day(self, date_str: str) -> Optional[MarketCalendar]:
        """Get calendar info for a specific date."""
        with self.session() as session:
            calendar = session.query(MarketCalendar).filter(
                MarketCalendar.date == date_str
            ).first()
            if calendar:
                session.expunge(calendar)
            return calendar

    def get_calendar_range(
        self,
        start_date: str,
        end_date: str,
    ) -> List[MarketCalendar]:
        """
        Get calendar info for a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of MarketCalendar records, sorted by date.
        """
        with self.session() as session:
            calendars = (
                session.query(MarketCalendar)
                .filter(MarketCalendar.date >= start_date)
                .filter(MarketCalendar.date <= end_date)
                .order_by(MarketCalendar.date)
                .all()
            )
            for cal in calendars:
                session.expunge(cal)
            return calendars

    def get_trading_days(
        self,
        start_date: str,
        end_date: str,
    ) -> List[str]:
        """
        Get list of trading days (open or early_close) in a date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of date strings for trading days.
        """
        with self.session() as session:
            calendars = (
                session.query(MarketCalendar.date)
                .filter(MarketCalendar.date >= start_date)
                .filter(MarketCalendar.date <= end_date)
                .filter(MarketCalendar.schedule_type.in_(["open", "early_close"]))
                .order_by(MarketCalendar.date)
                .all()
            )
            return [cal.date for cal in calendars]

    def is_trading_day(self, date_str: str) -> Optional[bool]:
        """
        Check if a date is a trading day.

        Returns:
            True if trading day, False if not, None if not in calendar.
        """
        calendar = self.get_calendar_day(date_str)
        if calendar is None:
            return None
        return calendar.is_trading_day

    def get_calendar_coverage(self) -> Dict[str, Any]:
        """
        Get info about calendar data coverage.

        Returns:
            Dict with min_date, max_date, total_days, trading_days, etc.
        """
        from sqlalchemy import func

        with self.session() as session:
            result = session.query(
                func.min(MarketCalendar.date),
                func.max(MarketCalendar.date),
                func.count(MarketCalendar.date),
            ).first()

            trading_count = session.query(func.count(MarketCalendar.date)).filter(
                MarketCalendar.schedule_type.in_(["open", "early_close"])
            ).scalar()

            return {
                "min_date": result[0],
                "max_date": result[1],
                "total_days": result[2],
                "trading_days": trading_count or 0,
            }

    def get_missing_calendar_dates(
        self,
        start_date: str,
        end_date: str,
    ) -> List[str]:
        """
        Find dates in range that are not in the calendar.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            List of date strings not in calendar.
        """
        from datetime import date as dt_date, timedelta

        # Get all dates in calendar for the range
        existing = set(
            cal.date for cal in self.get_calendar_range(start_date, end_date)
        )

        # Generate all dates in range
        start = dt_date.fromisoformat(start_date)
        end = dt_date.fromisoformat(end_date)
        all_dates = []
        current = start
        while current <= end:
            date_str = current.isoformat()
            if date_str not in existing:
                all_dates.append(date_str)
            current += timedelta(days=1)

        return all_dates

    # =========================================================================
    # Live Position State Operations
    # =========================================================================

    def get_position_state(self, session_id: str = "default") -> Optional[LivePositionState]:
        """
        Get live position state for a trading session.

        Args:
            session_id: Session identifier (default: "default")

        Returns:
            LivePositionState or None if not found.
        """
        with self.session() as session:
            state = (
                session.query(LivePositionState)
                .filter(LivePositionState.session_id == session_id)
                .first()
            )
            if state:
                session.expunge(state)
            return state

    def save_position_state(
        self,
        session_id: str = "default",
        **kwargs,
    ) -> LivePositionState:
        """
        Save or update live position state.

        Args:
            session_id: Session identifier
            **kwargs: Position state fields to update

        Returns:
            Updated LivePositionState.
        """
        with self.session() as session:
            state = (
                session.query(LivePositionState)
                .filter(LivePositionState.session_id == session_id)
                .first()
            )

            if state is None:
                state = LivePositionState(session_id=session_id)
                session.add(state)

            # Update all provided fields
            allowed_fields = {
                "has_active_position", "position_id", "position_type",
                "entry_time", "entry_option_price", "option_symbol",
                "strike", "num_contracts", "entry_underlying_price",
                "position_value", "dominant_model", "max_hold_minutes",
                "barrier_start_time", "confluence_count", "peak_pnl_pct",
                "breakeven_activated", "is_runner", "runner_start_time",
                "runner_max_hold_minutes", "runner_peak_pnl_pct",
                "runner_entry_pnl_pct", "is_breach", "is_reversal", "is_bounce", "renewals",
                "daily_trades", "daily_pnl", "capital", "last_trade_date",
                "completed_trades_json",
            }

            for key, value in kwargs.items():
                if key in allowed_fields:
                    setattr(state, key, value)

            state.updated_at = datetime.utcnow()
            session.flush()
            session.expunge(state)
            return state

    def clear_active_position(self, session_id: str = "default") -> None:
        """
        Clear the active position (set has_active_position to False).

        Args:
            session_id: Session identifier
        """
        with self.session() as session:
            state = (
                session.query(LivePositionState)
                .filter(LivePositionState.session_id == session_id)
                .first()
            )
            if state:
                state.has_active_position = False
                state.position_id = None
                state.updated_at = datetime.utcnow()

    def delete_position_state(self, session_id: str = "default") -> bool:
        """
        Delete position state for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found.
        """
        with self.session() as session:
            state = (
                session.query(LivePositionState)
                .filter(LivePositionState.session_id == session_id)
                .first()
            )
            if state:
                session.delete(state)
                return True
            return False


# Global database instance
_db_instance: Optional[Database] = None


def get_db() -> Database:
    """Get the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
        _db_instance.create_tables()
    return _db_instance


def init_db(database_url: Optional[str] = None) -> Database:
    """Initialize the database with a specific URL."""
    global _db_instance
    _db_instance = Database(database_url)
    _db_instance.create_tables()
    return _db_instance
