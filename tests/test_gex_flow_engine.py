"""
Unit tests for GEX Flow Engine module.

Tests Lee-Ready classifier and GEX flow feature computations.
"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from src.data.gex_flow_engine import (
    LeeReadyClassifier,
    GEXFlowEngine,
    GEXFlowFeatures,
)


class TestLeeReadyClassifier:
    """Tests for the Lee-Ready trade classification algorithm."""

    def test_buy_at_ask(self) -> None:
        """Trade at ask price should be classified as buy aggressor (sign=-1)."""
        classifier = LeeReadyClassifier()

        trade_quotes = pd.DataFrame({
            "trade_timestamp": [datetime(2024, 1, 15, 10, 0, 0)],
            "price": [5.00],
            "size": [10],
            "bid": [4.90],
            "ask": [5.00],
            "strike": [500.0],
            "right": ["call"],
        })

        result = classifier.classify_trades(trade_quotes)

        assert "sign" in result.columns
        assert result["sign"].iloc[0] == -1  # Buy at ask = dealer sold

    def test_sell_at_bid(self) -> None:
        """Trade at bid price should be classified as sell aggressor (sign=+1)."""
        classifier = LeeReadyClassifier()

        trade_quotes = pd.DataFrame({
            "trade_timestamp": [datetime(2024, 1, 15, 10, 0, 0)],
            "price": [4.90],
            "size": [10],
            "bid": [4.90],
            "ask": [5.00],
            "strike": [500.0],
            "right": ["call"],
        })

        result = classifier.classify_trades(trade_quotes)

        assert result["sign"].iloc[0] == 1  # Sell at bid = dealer bought

    def test_tick_rule_uptick(self) -> None:
        """Trade between bid/ask with uptick should be classified as buy."""
        classifier = LeeReadyClassifier()

        # Two trades: first at midpoint, second above first (uptick)
        trade_quotes = pd.DataFrame({
            "trade_timestamp": [
                datetime(2024, 1, 15, 10, 0, 0),
                datetime(2024, 1, 15, 10, 0, 1),
            ],
            "price": [4.94, 4.96],  # Both between bid/ask, second is uptick
            "size": [10, 10],
            "bid": [4.90, 4.90],
            "ask": [5.00, 5.00],
            "strike": [500.0, 500.0],
            "right": ["call", "call"],
        })

        result = classifier.classify_trades(trade_quotes)

        # Second trade is uptick -> buy -> sign = -1
        assert result["sign"].iloc[1] == -1

    def test_tick_rule_downtick(self) -> None:
        """Trade between bid/ask with downtick should be classified as sell."""
        classifier = LeeReadyClassifier()

        # Two trades: second below first (downtick)
        trade_quotes = pd.DataFrame({
            "trade_timestamp": [
                datetime(2024, 1, 15, 10, 0, 0),
                datetime(2024, 1, 15, 10, 0, 1),
            ],
            "price": [4.96, 4.94],  # Both between bid/ask, second is downtick
            "size": [10, 10],
            "bid": [4.90, 4.90],
            "ask": [5.00, 5.00],
            "strike": [500.0, 500.0],
            "right": ["call", "call"],
        })

        result = classifier.classify_trades(trade_quotes)

        # Second trade is downtick -> sell -> sign = +1
        assert result["sign"].iloc[1] == 1

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame should return empty DataFrame."""
        classifier = LeeReadyClassifier()

        trade_quotes = pd.DataFrame()
        result = classifier.classify_trades(trade_quotes)

        assert result.empty

    def test_batch_classification(self) -> None:
        """Test classification of multiple trades."""
        classifier = LeeReadyClassifier()

        trade_quotes = pd.DataFrame({
            "trade_timestamp": [
                datetime(2024, 1, 15, 10, 0, 0),
                datetime(2024, 1, 15, 10, 0, 1),
                datetime(2024, 1, 15, 10, 0, 2),
            ],
            "price": [5.00, 4.90, 4.95],  # At ask, at bid, midpoint
            "size": [10, 20, 15],
            "bid": [4.90, 4.90, 4.90],
            "ask": [5.00, 5.00, 5.00],
            "strike": [500.0, 500.0, 500.0],
            "right": ["call", "call", "call"],
        })

        result = classifier.classify_trades(trade_quotes)

        assert result["sign"].iloc[0] == -1  # At ask = buy
        assert result["sign"].iloc[1] == 1   # At bid = sell


class TestGEXFlowEngine:
    """Tests for the GEX Flow Engine."""

    def test_initialization(self) -> None:
        """Test engine initialization."""
        engine = GEXFlowEngine(strike_filter_pct=0.15, vwap_window_30m=30)

        assert engine.strike_filter_pct == 0.15
        assert engine.vwap_window_30m == 30
        assert engine._cumulative_net_gex == 0.0
        assert engine._cumulative_net_dex == 0.0

    def test_reset_daily_state(self) -> None:
        """Test daily state reset."""
        engine = GEXFlowEngine()
        engine._cumulative_net_gex = 1000.0
        engine._cumulative_net_dex = 500.0

        engine.reset_daily_state()

        assert engine._cumulative_net_gex == 0.0
        assert engine._cumulative_net_dex == 0.0

    def test_aggregate_trades_to_1min(self) -> None:
        """Test trade aggregation to 1-minute intervals."""
        engine = GEXFlowEngine()

        # Create classified trades
        trades = pd.DataFrame({
            "trade_timestamp": [
                datetime(2024, 1, 15, 10, 0, 15),
                datetime(2024, 1, 15, 10, 0, 30),
                datetime(2024, 1, 15, 10, 0, 45),
                datetime(2024, 1, 15, 10, 1, 15),
            ],
            "strike": [500.0, 500.0, 500.0, 500.0],
            "right": ["call", "call", "call", "call"],
            "size": [10, 20, 15, 25],
            "sign": [-1, 1, -1, 1],  # buy, sell, buy, sell
        })

        agg = engine._aggregate_trades_to_1min(trades)

        # First minute: buy=10+15=25, sell=20, net_vol = 20-25 = -5
        minute_1 = agg[agg["minute"] == datetime(2024, 1, 15, 10, 0)]
        assert len(minute_1) == 1
        assert minute_1["buy_volume"].iloc[0] == 25
        assert minute_1["sell_volume"].iloc[0] == 20
        assert minute_1["net_volume"].iloc[0] == -5

        # Second minute: buy=0, sell=25
        minute_2 = agg[agg["minute"] == datetime(2024, 1, 15, 10, 1)]
        assert len(minute_2) == 1
        assert minute_2["buy_volume"].iloc[0] == 0
        assert minute_2["sell_volume"].iloc[0] == 25

    def test_gamma_sentiment_ratio_bounds(self) -> None:
        """Test gamma sentiment ratio is bounded to [-1, 1]."""
        engine = GEXFlowEngine()

        # Normal case
        ratio = engine._compute_gamma_sentiment_ratio(100.0, 200.0)
        assert -1.0 <= ratio <= 1.0
        assert ratio == 0.5

        # Edge case: zero total flow
        ratio = engine._compute_gamma_sentiment_ratio(0.0, 0.0)
        assert ratio == 0.0

        # Edge case: net flow larger than total (shouldn't happen but handle gracefully)
        ratio = engine._compute_gamma_sentiment_ratio(300.0, 200.0)
        assert ratio == 1.0  # Clipped


class TestGEXFlowFeatures:
    """Tests for the GEXFlowFeatures dataclass."""

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        features = GEXFlowFeatures(
            timestamp=datetime(2024, 1, 15, 10, 0),
            underlying_price=500.0,
            total_gex=100000.0,
            net_gamma_flow=1000.0,
            cumulative_net_gex=5000.0,
            net_delta_flow=2000.0,
            anchored_vwap_z=1.5,
            gamma_sentiment_ratio=0.3,
            vwap_divergence=0.01,
            implied_volatility=0.25,
            pos_gex_wall_strike=505.0,
            neg_gex_wall_strike=495.0,
            zero_gex_price=500.0,
            zero_dex_price=500.5,
            gex_regime_strength=100000.0,
            call_buy_volume=500.0,
            call_sell_volume=300.0,
            put_buy_volume=400.0,
            put_sell_volume=200.0,
        )

        d = features.to_dict()

        assert d["timestamp"] == datetime(2024, 1, 15, 10, 0)
        assert d["underlying_price"] == 500.0
        assert d["net_gamma_flow"] == 1000.0
        assert d["gamma_sentiment_ratio"] == 0.3
        assert d["implied_volatility"] == 0.25
        assert d["gex_regime_strength"] == 100000.0
        assert d["call_buy_volume"] == 500.0
        assert d["put_sell_volume"] == 200.0
        assert len(d) == 19  # All 19 fields


class TestGEXFlowEngineIntegration:
    """Integration tests for GEX flow feature computation."""

    def test_compute_flow_features_empty_data(self) -> None:
        """Test handling of empty input data."""
        engine = GEXFlowEngine()

        result = engine.compute_flow_features(
            greeks_df=pd.DataFrame(),
            trade_agg_df=pd.DataFrame(),
            underlying_prices=pd.DataFrame(),
        )

        assert result.empty

    def test_compute_flow_features_no_trades(self) -> None:
        """Test computation with Greeks but no trades."""
        engine = GEXFlowEngine()

        greeks_df = pd.DataFrame({
            "timestamp": [datetime(2024, 1, 15, 10, 0)],
            "strike": [500.0],
            "right": ["call"],
            "delta": [0.5],
            "gamma": [0.02],
            "underlying_price": [500.0],
        })

        underlying_prices = pd.DataFrame({
            "timestamp": [datetime(2024, 1, 15, 10, 0)],
            "close": [500.0],
            "volume": [1000],
        })

        result = engine.compute_flow_features(
            greeks_df=greeks_df,
            trade_agg_df=pd.DataFrame(),
            underlying_prices=underlying_prices,
        )

        # Should compute features even without trades (flow = 0)
        assert len(result) == 1
        assert result["net_gamma_flow"].iloc[0] == 0.0
        assert result["net_delta_flow"].iloc[0] == 0.0
        assert "gex_regime_strength" in result.columns
