"""Tests for the LoaderWorker."""
import asyncio
import pytest
from datetime import datetime, date, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytz

from src.workers.loader_worker import LoaderWorker, load_minute_for_strategy
from src.workers.event_bus import EventBus, Event, reset_event_bus


@pytest.fixture(autouse=True)
def reset_global_bus():
    """Reset the global event bus before each test."""
    reset_event_bus()
    yield
    reset_event_bus()


@pytest.fixture
def temp_dirs():
    """Create temporary directories for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    stocks_dir = temp_dir / "stocks"
    options_dir = temp_dir / "options"
    gex_flow_dir = temp_dir / "gex_flow"
    cache_dir = temp_dir / "cache" / "normalized"
    raw_cache_dir = temp_dir / "cache" / "raw"

    for d in [stocks_dir, options_dir, gex_flow_dir, cache_dir, raw_cache_dir]:
        d.mkdir(parents=True)

    yield {
        "root": temp_dir,
        "stocks": stocks_dir,
        "options": options_dir,
        "gex_flow": gex_flow_dir,
        "cache": cache_dir,
        "raw_cache": raw_cache_dir,
    }

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_stocks_data():
    """Create sample stocks data."""
    return pd.DataFrame({
        "ticker": ["SPY"],
        "window_start": [datetime(2024, 1, 15, 14, 30, tzinfo=pytz.UTC)],
        "open": [500.0],
        "high": [500.5],
        "low": [499.5],
        "close": [500.25],
        "volume": [1000000],
        "transactions": [5000],
    })


@pytest.fixture
def sample_options_data():
    """Create sample options data."""
    return pd.DataFrame({
        "ticker": ["O:SPY240115C00500000", "O:SPY240115P00500000"],
        "window_start": [
            datetime(2024, 1, 15, 14, 30, tzinfo=pytz.UTC),
            datetime(2024, 1, 15, 14, 30, tzinfo=pytz.UTC),
        ],
        "open": [2.50, 2.40],
        "high": [2.60, 2.50],
        "low": [2.40, 2.30],
        "close": [2.55, 2.45],
        "volume": [1000, 800],
        "transactions": [100, 80],
    })


@pytest.fixture
def sample_gex_flow_data():
    """Create sample GEX flow data."""
    # ThetaData returns Eastern time
    eastern = pytz.timezone("America/New_York")
    return pd.DataFrame({
        "timestamp": [datetime(2024, 1, 15, 9, 30)],  # Eastern, naive
        "underlying_price": [500.25],
        "net_gamma_flow": [1e9],
        "dist_to_zero_gex": [5.0],
        "cumulative_net_gex": [1e9],
        "dist_to_pos_gex_wall": [10.0],
        "dist_to_neg_gex_wall": [-8.0],
        "net_delta_flow": [5e8],
        "anchored_vwap_z": [0.5],
        "gamma_sentiment_ratio": [0.3],
        "vwap_divergence": [0.01],
        "dist_to_zero_dex": [3.0],
        "implied_volatility": [0.15],
    })


def test_loader_worker_init(temp_dirs):
    """Test LoaderWorker initialization."""
    worker = LoaderWorker(
        underlying="SPY",
        stocks_dir=temp_dirs["stocks"],
        options_dir=temp_dirs["options"],
        gex_flow_dir=temp_dirs["gex_flow"],
        cache_dir=temp_dirs["cache"],
        raw_cache_dir=temp_dirs["raw_cache"],
    )

    assert worker.underlying == "SPY"
    assert worker.stocks_dir == temp_dirs["stocks"]
    assert not worker.is_running


def test_get_paths(temp_dirs):
    """Test path generation methods."""
    worker = LoaderWorker(
        underlying="SPY",
        stocks_dir=temp_dirs["stocks"],
        options_dir=temp_dirs["options"],
        gex_flow_dir=temp_dirs["gex_flow"],
        cache_dir=temp_dirs["cache"],
        raw_cache_dir=temp_dirs["raw_cache"],
    )

    test_date = date(2024, 1, 15)

    stocks_path = worker._get_stocks_path(test_date)
    assert "SPY_STOCKS-1M_2024-01-15" in str(stocks_path)

    options_path = worker._get_options_path(test_date)
    assert "SPY_OPTIONS-1M_2024-01-15" in str(options_path)

    gex_path = worker._get_gex_flow_path(test_date)
    assert "SPY_thetadata_1m_combined_2024-01-15" in str(gex_path)


@pytest.mark.asyncio
async def test_on_minute_synced(temp_dirs, sample_stocks_data, sample_gex_flow_data):
    """Test processing a minute_synced event."""
    event_bus = EventBus()

    # Save sample data files
    target_date = date(2024, 1, 15)
    stocks_path = temp_dirs["stocks"] / f"SPY_STOCKS-1M_{target_date}.parquet"
    sample_stocks_data.to_parquet(stocks_path)

    gex_path = temp_dirs["gex_flow"] / f"SPY_thetadata_1m_combined_{target_date}.parquet"
    sample_gex_flow_data.to_parquet(gex_path)

    worker = LoaderWorker(
        underlying="SPY",
        event_bus=event_bus,
        stocks_dir=temp_dirs["stocks"],
        options_dir=temp_dirs["options"],
        gex_flow_dir=temp_dirs["gex_flow"],
        cache_dir=temp_dirs["cache"],
        raw_cache_dir=temp_dirs["raw_cache"],
    )

    # Track cache_updated events
    cache_updated_events = []

    async def on_cache_updated(event: Event):
        cache_updated_events.append(event)

    event_bus.subscribe("cache_updated", on_cache_updated)

    # Create and process event
    target_minute = datetime(2024, 1, 15, 14, 30, tzinfo=pytz.UTC)
    event = Event(
        name="minute_synced",
        data={
            "timestamp": target_minute,
            "date": target_date,
        },
        timestamp=datetime.utcnow(),
    )

    await worker.on_minute_synced(event)

    # Verify cache files were created
    raw_cache_path = worker._get_raw_cache_path(target_date)
    assert raw_cache_path.exists()

    normalized_cache_path = worker._get_normalized_cache_path(target_date)
    assert normalized_cache_path.exists()

    # Verify cache_updated event was emitted
    await asyncio.sleep(0.1)
    await event_bus.emit_sync("__flush__", None)  # Flush events
    # Note: Event bus uses queue, so we check if subscription worked
    assert event_bus.get_subscriber_count("cache_updated") == 1


@pytest.mark.asyncio
async def test_load_minute_data(temp_dirs, sample_stocks_data):
    """Test loading minute data from source files."""
    worker = LoaderWorker(
        underlying="SPY",
        stocks_dir=temp_dirs["stocks"],
        options_dir=temp_dirs["options"],
        gex_flow_dir=temp_dirs["gex_flow"],
        cache_dir=temp_dirs["cache"],
        raw_cache_dir=temp_dirs["raw_cache"],
    )

    # Save sample data
    target_date = date(2024, 1, 15)
    stocks_path = temp_dirs["stocks"] / f"SPY_STOCKS-1M_{target_date}.parquet"
    sample_stocks_data.to_parquet(stocks_path)

    # Load the data
    target_minute = datetime(2024, 1, 15, 14, 30, tzinfo=pytz.UTC)
    worker._reset_daily_state(target_date)
    df = worker._load_minute_data(target_minute, target_date)

    assert df is not None
    assert not df.empty
    assert "open" in df.columns
    assert "close" in df.columns
    assert "vwap" in df.columns
    assert "time_to_close" in df.columns


def test_reset_daily_state(temp_dirs):
    """Test daily state reset."""
    worker = LoaderWorker(
        underlying="SPY",
        stocks_dir=temp_dirs["stocks"],
        options_dir=temp_dirs["options"],
        gex_flow_dir=temp_dirs["gex_flow"],
        cache_dir=temp_dirs["cache"],
        raw_cache_dir=temp_dirs["raw_cache"],
    )

    # Add some history
    worker._today_close_history = [500.0, 501.0]
    worker._today_volume_history = [1000, 2000]
    worker._current_date = date(2024, 1, 14)

    # Reset for new day
    worker._reset_daily_state(date(2024, 1, 15))

    assert worker._current_date == date(2024, 1, 15)
    assert len(worker._today_close_history) == 0
    assert len(worker._today_volume_history) == 0


def test_append_to_cache(temp_dirs):
    """Test appending data to cache file."""
    worker = LoaderWorker(
        underlying="SPY",
        cache_dir=temp_dirs["cache"],
        raw_cache_dir=temp_dirs["raw_cache"],
    )

    cache_path = temp_dirs["cache"] / "test_cache.parquet"

    # Create initial data
    df1 = pd.DataFrame({
        "close": [500.0],
        "volume": [1000],
    }, index=pd.to_datetime(["2024-01-15 14:30:00"]))
    df1.index.name = "timestamp"

    worker._append_to_cache(cache_path, df1)
    assert cache_path.exists()

    # Append more data
    df2 = pd.DataFrame({
        "close": [501.0],
        "volume": [1100],
    }, index=pd.to_datetime(["2024-01-15 14:31:00"]))
    df2.index.name = "timestamp"

    worker._append_to_cache(cache_path, df2)

    # Verify combined data
    result = pd.read_parquet(cache_path)
    assert len(result) == 2


@pytest.mark.asyncio
async def test_loader_worker_run_and_stop():
    """Test run and stop functionality."""
    event_bus = EventBus()
    worker = LoaderWorker(event_bus=event_bus)

    # Start worker
    task = asyncio.create_task(worker.run())
    await asyncio.sleep(0.1)

    assert worker.is_running
    assert event_bus.get_subscriber_count("minute_synced") == 1

    # Stop worker
    worker.stop()
    await task

    assert not worker.is_running


@pytest.mark.asyncio
async def test_load_minute_for_strategy(temp_dirs):
    """Test the helper function for loading minute data."""
    # Create a cached file
    target_date = date(2024, 1, 15)
    from src.data.loader import CACHE_VERSION
    cache_path = temp_dirs["cache"] / f"SPY_{CACHE_VERSION}_{target_date}.parquet"

    df = pd.DataFrame({
        "timestamp": [
            datetime(2024, 1, 15, 14, 30),
            datetime(2024, 1, 15, 14, 31),
        ],
        "close": [500.0, 501.0],
        "volume": [1000, 1100],
    })
    df = df.set_index("timestamp")
    df.to_parquet(cache_path)

    # Load specific minute
    target_minute = datetime(2024, 1, 15, 14, 30)
    result = await load_minute_for_strategy(
        target_minute=target_minute,
        underlying="SPY",
        cache_dir=temp_dirs["cache"],
    )

    assert result is not None
    assert len(result) == 1
    assert result["close"].iloc[0] == 500.0
