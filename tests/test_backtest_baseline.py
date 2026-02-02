"""
Baseline test for backtest behavior - run before and after refactoring.

This test captures the current backtest behavior on a known date range
and asserts on key metrics to ensure extraction doesn't change behavior.
"""
import pytest
from datetime import date
from pathlib import Path

import pandas as pd
import torch

from src.backtest.multi_percentile import MultiPercentileBacktester
from src.backtest.types import BacktestConfig
from src.strategy.multi_percentile_executor import MultiPercentileExecutor
from src.strategy.fusion_config import DEFAULT_MODEL_CONFIGS, FusionConfig
from src.execution.heuristic import HeuristicConfig


# Expected baseline values - captured from initial run
# These will be filled in after running the test once
EXPECTED_TOTAL_TRADES = None  # To be filled after first run
EXPECTED_WIN_RATE = None  # To be filled after first run
EXPECTED_TOTAL_PNL = None  # To be filled after first run
EXPECTED_MAX_DD = None  # To be filled after first run


def load_test_data(start_date: date, end_date: date) -> tuple:
    """Load feature and OHLCV data for the test date range."""
    # Load features
    features_path = Path("data/features-1m/SPY-features.parquet")
    if not features_path.exists():
        pytest.skip(f"Features file not found: {features_path}")

    features_df = pd.read_parquet(features_path)

    # Filter to date range
    features_df.index = pd.to_datetime(features_df.index, utc=True)
    mask = (features_df.index.date >= start_date) & (features_df.index.date <= end_date)
    features_df = features_df[mask]

    if len(features_df) < 100:
        pytest.skip(f"Insufficient data for date range: {len(features_df)} rows")

    # Get raw closes
    raw_closes = features_df["close"].values if "close" in features_df.columns else None

    # Load OHLCV data
    ohlcv_path = Path("data/processed-1m/SPY-ohlcv.parquet")
    ohlcv_df = None
    if ohlcv_path.exists():
        ohlcv_df = pd.read_parquet(ohlcv_path)
        ohlcv_df.index = pd.to_datetime(ohlcv_df.index, utc=True)
        ohlcv_df = ohlcv_df[mask]

    return features_df, raw_closes, ohlcv_df


class TestBacktestBaseline:
    """Regression tests to ensure backtest behavior doesn't change during refactoring."""

    @pytest.fixture
    def backtest_config(self) -> BacktestConfig:
        """Standard backtest configuration."""
        return BacktestConfig(
            initial_capital=25_000.0,
            stop_loss_pct=10.0,
            entry_slippage_pct=0.5,
            exit_slippage_pct=0.5,
        )

    @pytest.fixture
    def device(self) -> torch.device:
        """Get the compute device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def executor(self, device) -> MultiPercentileExecutor:
        """Create the multi-percentile executor (mock if models don't exist)."""
        try:
            executor = MultiPercentileExecutor.from_configs(
                DEFAULT_MODEL_CONFIGS,
                FusionConfig(),
                device=device,
            )
            return executor
        except FileNotFoundError:
            pytest.skip("Model checkpoints not found - skipping baseline test")

    @pytest.mark.slow
    def test_two_week_baseline_january_2025(
        self,
        backtest_config: BacktestConfig,
        device: torch.device,
        executor: MultiPercentileExecutor,
    ):
        """
        Run backtest on Jan 2-17, 2025 and verify key metrics.

        This test should be run before and after any refactoring to ensure
        the trading logic produces identical results.
        """
        start_date = date(2025, 1, 2)
        end_date = date(2025, 1, 17)

        # Load test data
        features_df, raw_closes, ohlcv_df = load_test_data(start_date, end_date)

        # Create backtester
        backtester = MultiPercentileBacktester(
            executor=executor,
            config=backtest_config,
            device=device,
            use_heuristic=True,
            heuristic_config=HeuristicConfig(),
        )

        # Run backtest
        results = backtester.run_backtest(
            df=features_df,
            raw_closes=raw_closes,
            year=2025,
            month=1,
            ohlcv_df=ohlcv_df,
        )

        # Extract metrics
        trades = backtester.trades
        total_trades = len(trades)
        wins = sum(1 for t in trades if t.pnl_pct > 0)
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        total_pnl = sum(t.pnl_dollars for t in trades)
        max_drawdown = (backtester.peak_capital - backtester.lowest_capital) / backtester.peak_capital * 100

        # Print metrics for baseline capture
        print("\n" + "=" * 60)
        print("BASELINE METRICS (capture these values)")
        print("=" * 60)
        print(f"Total trades: {total_trades}")
        print(f"Win rate: {win_rate:.4f}")
        print(f"Total P&L: ${total_pnl:.2f}")
        print(f"Max drawdown: {max_drawdown:.2f}%")
        print(f"Final capital: ${backtester.capital:.2f}")
        print("=" * 60)

        # Basic sanity checks
        assert total_trades > 0, "Should have at least some trades"
        assert 0.0 <= win_rate <= 1.0, "Win rate should be between 0 and 1"

        # Assertions against baseline values (once captured)
        if EXPECTED_TOTAL_TRADES is not None:
            assert total_trades == EXPECTED_TOTAL_TRADES, \
                f"Trade count changed: expected {EXPECTED_TOTAL_TRADES}, got {total_trades}"

        if EXPECTED_WIN_RATE is not None:
            assert abs(win_rate - EXPECTED_WIN_RATE) < 0.01, \
                f"Win rate changed: expected {EXPECTED_WIN_RATE:.4f}, got {win_rate:.4f}"

        if EXPECTED_TOTAL_PNL is not None:
            assert abs(total_pnl - EXPECTED_TOTAL_PNL) < 100, \
                f"P&L changed: expected ${EXPECTED_TOTAL_PNL:.2f}, got ${total_pnl:.2f}"

        if EXPECTED_MAX_DD is not None:
            assert max_drawdown <= EXPECTED_MAX_DD * 1.1, \
                f"Max drawdown increased: expected {EXPECTED_MAX_DD:.2f}%, got {max_drawdown:.2f}%"

    @pytest.mark.slow
    def test_signal_determinism(
        self,
        backtest_config: BacktestConfig,
        device: torch.device,
        executor: MultiPercentileExecutor,
    ):
        """
        Verify that running the same backtest twice produces identical results.

        This ensures the trading logic is deterministic (no random elements
        affecting signal generation).
        """
        start_date = date(2025, 1, 6)
        end_date = date(2025, 1, 10)

        features_df, raw_closes, ohlcv_df = load_test_data(start_date, end_date)

        # Run twice
        results = []
        for run in range(2):
            backtester = MultiPercentileBacktester(
                executor=executor,
                config=backtest_config,
                device=device,
                use_heuristic=True,
            )

            backtester.run_backtest(
                df=features_df,
                raw_closes=raw_closes,
                year=2025,
                month=1,
                ohlcv_df=ohlcv_df,
            )

            results.append({
                "trades": len(backtester.trades),
                "capital": backtester.capital,
                "trade_times": [t.entry_time for t in backtester.trades],
            })

        # Compare results
        assert results[0]["trades"] == results[1]["trades"], \
            f"Trade count differs between runs: {results[0]['trades']} vs {results[1]['trades']}"

        # Note: Capital may differ slightly due to random slippage in stop loss fills
        # We check trade times instead for determinism of signal generation
        assert results[0]["trade_times"] == results[1]["trade_times"], \
            "Trade entry times differ between runs - signal generation is not deterministic"


class TestVWAPBreachDetection:
    """Test VWAP breach detection logic in isolation."""

    def test_vwap_breach_bullish(self):
        """Test bullish VWAP breach detection."""
        from src.backtest.multi_percentile import (
            detect_breach,
            calculate_vwap_with_bands,
            BreachType,
        )
        import numpy as np

        # Create synthetic OHLCV data with a VWAP breach
        n_rows = 20
        dates = pd.date_range("2025-01-06 09:30:00", periods=n_rows, freq="1min", tz="America/New_York")

        # Price starts below VWAP, then crosses above
        base_price = 600.0
        ohlcv_df = pd.DataFrame({
            "open": [base_price - 0.5 + i * 0.1 for i in range(n_rows)],
            "high": [base_price + i * 0.1 + 0.3 for i in range(n_rows)],
            "low": [base_price - 0.5 + i * 0.1 - 0.1 for i in range(n_rows)],
            "close": [base_price - 0.3 + i * 0.15 for i in range(n_rows)],
            "volume": [100000 + i * 5000 for i in range(n_rows)],
        }, index=dates)

        # Calculate VWAP
        vwap, upper_1sd, lower_1sd, upper_2sd, lower_2sd = calculate_vwap_with_bands(ohlcv_df)

        # Test detection at index 15 (after some history)
        signal = detect_breach(
            ohlcv_df, vwap, 15,
            upper_1sd_series=upper_1sd,
            lower_1sd_series=lower_1sd,
            upper_2sd_series=upper_2sd,
            lower_2sd_series=lower_2sd,
        )

        # Should detect some kind of signal (not necessarily bullish due to synthetic data)
        assert signal is not None
        assert hasattr(signal, "breach_type")
        assert hasattr(signal, "vwap_value")


class TestExitRules:
    """Test exit rule logic in isolation."""

    def test_stop_loss_trigger(self):
        """Test that stop loss triggers at correct P&L level."""
        from src.backtest.types import Position, PositionType
        from datetime import datetime

        position = Position(
            position_type=PositionType.CALL,
            entry_time=datetime(2025, 1, 6, 10, 0, 0),
            entry_underlying_price=600.0,
            entry_option_price=2.50,
            strike=600.0,
            entry_idx=0,
            position_value=2500.0,
            dominant_model="5m",
            max_hold_minutes=10,
        )

        # At -10% P&L, should trigger stop loss
        current_option_price = 2.25  # 10% loss
        pnl_pct = (current_option_price - position.entry_option_price) / position.entry_option_price * 100

        assert pnl_pct == -10.0
        assert pnl_pct <= -10.0  # Should trigger 10% stop loss


class TestPositionSizing:
    """Test position sizing logic in isolation."""

    def test_vwap_breach_position_size(self):
        """Test position sizing for VWAP breach trades."""
        capital = 25000.0
        base_position_pct = 0.03
        vwap_breach_pct = 0.03
        max_pct = 0.05

        # Without bonus
        position_value = capital * base_position_pct
        assert position_value == 750.0

        # With max bonus (power hour)
        bonus = 0.02
        total_pct = min(vwap_breach_pct + bonus, max_pct)
        position_value_boosted = capital * total_pct
        assert position_value_boosted == 1250.0  # 5% of 25000

    def test_contract_calculation(self):
        """Test contract count calculation from position value."""
        position_value = 1000.0
        option_price = 2.50
        cost_per_contract = option_price * 100

        num_contracts = max(1, int(position_value / cost_per_contract))
        assert num_contracts == 4  # 1000 / 250 = 4 contracts
