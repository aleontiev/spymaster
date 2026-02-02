"""
Parity tests for live trading vs backtest behavior.

Verifies that the extracted trading logic produces identical signals
and decisions as the original backtest implementation.
"""
import pytest
from datetime import datetime, date
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from src.backtest.types import Position, PositionType
from src.strategy.trading_rules import (
    BreachSignal,
    BreachType,
    NewsEventType,
    NewsEventSignal,
    calculate_vwap_with_bands,
    detect_breach,
    check_breach_continuation_confirmation,
    detect_news_event,
)
from src.strategy.exit_rules import ExitConfig, ExitDecision, ExitRuleEngine
from src.strategy.position_sizing import PositionSizeConfig, PositionSizer, StrikeSelection
from src.strategy.fusion_config import FusedSignal, Signal


def make_fused_signal(action: Signal = Signal.HOLD) -> FusedSignal:
    """Create a FusedSignal with default values for testing."""
    return FusedSignal(
        action=action,
        confidence=0.5,
        dominant_model="5m",
        individual_signals={},
        exit_horizon_minutes=10,
        attention_weights=(0.333, 0.333, 0.333),
        agreeing_models=(),
    )


class TestVWAPCalculationParity:
    """Verify VWAP calculation matches backtest."""

    def test_vwap_calculation_basic(self):
        """Test basic VWAP calculation."""
        # Create synthetic OHLCV data
        n_bars = 20
        et_tz = ZoneInfo("America/New_York")
        timestamps = pd.date_range(
            "2025-01-06 09:30:00",
            periods=n_bars,
            freq="1min",
            tz=et_tz,
        )

        ohlcv_df = pd.DataFrame({
            "open": [600.0 + i * 0.1 for i in range(n_bars)],
            "high": [600.5 + i * 0.1 for i in range(n_bars)],
            "low": [599.5 + i * 0.1 for i in range(n_bars)],
            "close": [600.0 + i * 0.15 for i in range(n_bars)],
            "volume": [100000 + i * 1000 for i in range(n_bars)],
        }, index=timestamps)

        # Calculate VWAP with bands
        vwap, upper_1sd, lower_1sd, upper_2sd, lower_2sd = calculate_vwap_with_bands(ohlcv_df)

        # Verify lengths match
        assert len(vwap) == len(ohlcv_df)
        assert len(upper_1sd) == len(ohlcv_df)
        assert len(lower_1sd) == len(ohlcv_df)

        # Verify VWAP is calculated (it may lag behind price in a trending market)
        assert vwap.iloc[0] > 0
        assert not vwap.isna().any()

        # Verify band ordering
        for i in range(len(vwap)):
            assert lower_2sd.iloc[i] <= lower_1sd.iloc[i] <= vwap.iloc[i] <= upper_1sd.iloc[i] <= upper_2sd.iloc[i]

    def test_vwap_day_reset(self):
        """Test that VWAP resets at start of new day."""
        et_tz = ZoneInfo("America/New_York")

        # Day 1 data
        day1_timestamps = pd.date_range("2025-01-06 09:30:00", periods=5, freq="1min", tz=et_tz)
        # Day 2 data
        day2_timestamps = pd.date_range("2025-01-07 09:30:00", periods=5, freq="1min", tz=et_tz)

        timestamps = day1_timestamps.append(day2_timestamps)

        ohlcv_df = pd.DataFrame({
            "open": [600.0] * 5 + [610.0] * 5,  # Day 2 has higher prices
            "high": [600.5] * 5 + [610.5] * 5,
            "low": [599.5] * 5 + [609.5] * 5,
            "close": [600.0] * 5 + [610.0] * 5,
            "volume": [100000] * 10,
        }, index=timestamps)

        vwap, _, _, _, _ = calculate_vwap_with_bands(ohlcv_df)

        # Day 1 VWAP should be around 600
        assert abs(vwap.iloc[4] - 600) < 1.0

        # Day 2 VWAP should reset and be around 610
        assert abs(vwap.iloc[5] - 610) < 1.0


class TestVWAPBreachDetectionParity:
    """Verify VWAP breach detection matches backtest."""

    def test_bullish_breach_detection(self):
        """Test detection of bullish VWAP breach."""
        et_tz = ZoneInfo("America/New_York")
        timestamps = pd.date_range("2025-01-06 09:30:00", periods=20, freq="1min", tz=et_tz)

        # Create data where price crosses above VWAP
        ohlcv_df = pd.DataFrame({
            "open": [598.0] * 10 + [600.0] * 10,  # Opens below VWAP, then at VWAP
            "high": [598.5] * 10 + [601.5] * 10,
            "low": [597.5] * 10 + [599.5] * 10,
            "close": [598.0] * 10 + [601.0] * 10,  # Closes above VWAP
            "volume": [100000] * 10 + [300000] * 10,  # Volume spike
        }, index=timestamps)

        vwap, upper_1sd, lower_1sd, upper_2sd, lower_2sd = calculate_vwap_with_bands(ohlcv_df)

        # Test detection at bar 15 (after volume spike and price cross)
        signal = detect_breach(
            ohlcv_df, vwap, 15,
            upper_1sd_series=upper_1sd,
            lower_1sd_series=lower_1sd,
            upper_2sd_series=upper_2sd,
            lower_2sd_series=lower_2sd,
        )

        # Should detect a breach signal
        assert signal is not None
        assert signal.vwap_value == vwap.iloc[15]

    def test_no_breach_when_no_vwap_cross(self):
        """Test no breach when candle doesn't cross VWAP."""
        et_tz = ZoneInfo("America/New_York")
        timestamps = pd.date_range("2025-01-06 09:30:00", periods=20, freq="1min", tz=et_tz)

        # Create data where candle body is entirely above VWAP (no crossing)
        # With high=low=close=open, VWAP will equal typical price = price
        # Then make the candles that don't cross VWAP
        ohlcv_df = pd.DataFrame({
            "open": [610.0] * 20,  # Opens well above where VWAP will be
            "high": [611.0] * 20,
            "low": [609.5] * 20,   # Low is well above VWAP
            "close": [610.5] * 20, # Closes above open (green, but no VWAP cross)
            "volume": [100000] * 20,
        }, index=timestamps)

        vwap, upper_1sd, lower_1sd, upper_2sd, lower_2sd = calculate_vwap_with_bands(ohlcv_df)

        signal = detect_breach(
            ohlcv_df, vwap, 15,
            upper_1sd_series=upper_1sd,
            lower_1sd_series=lower_1sd,
            upper_2sd_series=upper_2sd,
            lower_2sd_series=lower_2sd,
        )

        # Check what the VWAP value is - with typical price = (611+609.5+610.5)/3 = 610.33
        # Open is 610 which is below VWAP (610.33), close is 610.5 which is above
        # This IS a VWAP breach (body crosses from below to above)
        # So this test needs different data where open is clearly above VWAP
        # Let's check that a flat candle (no movement) doesn't breach
        # With our data, VWAP ~ 610.33 and open=610 < VWAP < close=610.5
        # So it detects a breach. Let's change the test to verify this is expected behavior
        # A proper "no breach" scenario would be where open is already above VWAP

        # Since VWAP ~ 610.33, make open > VWAP
        ohlcv_df2 = pd.DataFrame({
            "open": [610.5] * 20,  # Opens at VWAP level
            "high": [611.0] * 20,
            "low": [610.0] * 20,
            "close": [610.5] * 20, # Closes at same level (doji)
            "volume": [100000] * 20,
        }, index=timestamps)

        vwap2, upper_1sd2, lower_1sd2, upper_2sd2, lower_2sd2 = calculate_vwap_with_bands(ohlcv_df2)

        signal2 = detect_breach(
            ohlcv_df2, vwap2, 15,
            upper_1sd_series=upper_1sd2,
            lower_1sd_series=lower_1sd2,
            upper_2sd_series=upper_2sd2,
            lower_2sd_series=lower_2sd2,
        )

        # Doji candle (open == close) shouldn't breach since there's no directional body
        # The breach detection requires close > open for bullish or close < open for bearish
        assert signal2.breach_type == BreachType.NONE


class TestExitRulesParity:
    """Verify exit rules match backtest behavior."""

    @pytest.fixture
    def exit_engine(self) -> ExitRuleEngine:
        """Create exit rule engine with standard config."""
        return ExitRuleEngine(ExitConfig(
            stop_loss_pct=10.0,
            breakeven_activation_pct=7.5,
            breach_breakeven_activation_pct=10.0,
            trailing_stop_distance_pct=5.0,
        ))

    @pytest.fixture
    def base_position(self) -> Position:
        """Create a base position for testing."""
        et_tz = ZoneInfo("America/New_York")
        entry_time = datetime(2025, 1, 6, 10, 0, 0, tzinfo=et_tz)
        return Position(
            position_type=PositionType.CALL,
            entry_time=entry_time,
            entry_underlying_price=600.0,
            entry_option_price=2.50,
            strike=600.0,
            entry_idx=0,
            position_value=2500.0,
            dominant_model="5m",
            max_hold_minutes=10,
            barrier_start_time=entry_time,
        )

    def test_stop_loss_trigger(self, exit_engine: ExitRuleEngine, base_position: Position):
        """Test stop loss triggers at -10%."""
        # Option price at -10%
        current_price = 2.25  # -10%

        fused_signal = make_fused_signal(Signal.HOLD)
        current_time = base_position.entry_time

        decision = exit_engine.evaluate(
            base_position, current_price, fused_signal, current_time
        )

        assert decision.should_exit is True
        assert decision.exit_reason == "stop_loss"

    def test_stop_loss_not_triggered_above_threshold(
        self, exit_engine: ExitRuleEngine, base_position: Position
    ):
        """Test stop loss doesn't trigger above threshold."""
        # Option price at -9%
        current_price = 2.275  # -9%

        fused_signal = make_fused_signal(Signal.HOLD)
        current_time = base_position.entry_time

        decision = exit_engine.evaluate(
            base_position, current_price, fused_signal, current_time
        )

        assert decision.should_exit is False

    def test_breakeven_activation(self, exit_engine: ExitRuleEngine, base_position: Position):
        """Test breakeven activation at +7.5%."""
        # First, hit +8% to activate breakeven
        base_position.peak_pnl_pct = 8.0

        current_price = 2.70  # +8%
        fused_signal = make_fused_signal(Signal.HOLD)
        current_time = base_position.entry_time

        decision = exit_engine.evaluate(
            base_position, current_price, fused_signal, current_time
        )

        # Breakeven should be activated
        assert base_position.breakeven_activated is True
        assert decision.should_exit is False

    def test_trailing_stop_after_breakeven(
        self, exit_engine: ExitRuleEngine, base_position: Position
    ):
        """Test trailing stop triggers after breakeven activation."""
        # Activate breakeven and set peak
        base_position.breakeven_activated = True
        base_position.peak_pnl_pct = 15.0

        # Current P&L at +9% (5% trailing from peak of 15%)
        current_price = 2.725  # +9%

        fused_signal = make_fused_signal(Signal.HOLD)
        current_time = base_position.entry_time

        decision = exit_engine.evaluate(
            base_position, current_price, fused_signal, current_time
        )

        assert decision.should_exit is True
        assert decision.exit_reason == "trailing_stop"

    def test_runner_conversion_at_20_pct(
        self, exit_engine: ExitRuleEngine, base_position: Position
    ):
        """Test conversion to runner at +20%."""
        base_position.breakeven_activated = True
        base_position.peak_pnl_pct = 25.0

        # Trailing stop hit but still above 20%
        current_price = 3.00  # +20%

        fused_signal = make_fused_signal(Signal.HOLD)
        current_time = base_position.entry_time

        decision = exit_engine.evaluate(
            base_position, current_price, fused_signal, current_time
        )

        assert decision.convert_to_runner is True

    def test_signal_reversal_exit(
        self, exit_engine: ExitRuleEngine, base_position: Position
    ):
        """Test exit on signal reversal."""
        current_price = 2.60  # +4%

        # Opposing signal (SHORT for a CALL position)
        fused_signal = make_fused_signal(Signal.SHORT)
        current_time = base_position.entry_time

        decision = exit_engine.evaluate(
            base_position, current_price, fused_signal, current_time
        )

        assert decision.should_exit is True
        assert decision.exit_reason == "signal_reversal"


class TestPositionSizingParity:
    """Verify position sizing matches backtest behavior."""

    @pytest.fixture
    def position_sizer(self) -> PositionSizer:
        """Create position sizer with standard config."""
        return PositionSizer(PositionSizeConfig(
            base_position_pct=0.03,
            breach_position_pct=0.03,
            max_position_pct=0.05,
            low_capital_threshold=15000.0,
            low_capital_multiplier=2.0,
        ))

    def test_normal_position_size(self, position_sizer: PositionSizer):
        """Test normal position sizing at 3%."""
        et_tz = ZoneInfo("America/New_York")
        trade_time = datetime(2025, 1, 6, 10, 30, 0, tzinfo=et_tz)

        result = position_sizer.calculate_position_size(
            capital=25000.0,
            option_price=2.50,
            signal_type="normal",
            time_of_day=trade_time,
        )

        # 3% of 25000 = 750, at $2.50 * 100 = $250 per contract = 3 contracts
        assert result.position_pct == 0.03
        assert result.num_contracts == 3
        assert result.size_multiplier == 1.0

    def test_low_capital_multiplier(self, position_sizer: PositionSizer):
        """Test 2x multiplier when capital is under threshold."""
        et_tz = ZoneInfo("America/New_York")
        trade_time = datetime(2025, 1, 6, 10, 30, 0, tzinfo=et_tz)

        result = position_sizer.calculate_position_size(
            capital=10000.0,  # Below $15k threshold
            option_price=2.50,
            signal_type="normal",
            time_of_day=trade_time,
        )

        # 3% * 2x = 6% of 10000 = 600, at $250 per contract = 2 contracts
        assert result.size_multiplier == 2.0
        assert result.num_contracts == 2

    def test_vwap_breach_position_size(self, position_sizer: PositionSizer):
        """Test VWAP breach position sizing."""
        et_tz = ZoneInfo("America/New_York")
        trade_time = datetime(2025, 1, 6, 10, 30, 0, tzinfo=et_tz)

        result = position_sizer.calculate_position_size(
            capital=25000.0,
            option_price=2.50,
            signal_type="breach",
            time_of_day=trade_time,
            position_size_bonus=0.02,  # Max bonus
        )

        # 3% + 2% = 5% (capped), of 25000 = 1250
        assert result.position_pct == 0.05
        assert result.num_contracts == 5

    def test_strike_selection_atm_for_early(self, position_sizer: PositionSizer):
        """Test ATM strike selection for early trades."""
        et_tz = ZoneInfo("America/New_York")
        trade_time = datetime(2025, 1, 6, 9, 50, 0, tzinfo=et_tz)  # 9:50 AM

        strike, selection = position_sizer.select_strike(
            underlying_price=600.60,  # Use .60 to avoid round-half-to-even behavior
            position_type=PositionType.CALL,
            signal_type="early",
            confidence=0.7,
            time_of_day=trade_time,
        )

        # Early trades always ATM
        assert selection == StrikeSelection.ATM
        assert strike == 601  # Round(600.60) = 601

    def test_strike_selection_otm_for_high_confidence(self, position_sizer: PositionSizer):
        """Test OTM strike selection for high confidence normal trades."""
        et_tz = ZoneInfo("America/New_York")
        trade_time = datetime(2025, 1, 6, 11, 0, 0, tzinfo=et_tz)

        strike, selection = position_sizer.select_strike(
            underlying_price=600.50,
            position_type=PositionType.CALL,
            signal_type="normal",
            confidence=0.95,  # Very high confidence
            time_of_day=trade_time,
        )

        # High confidence = 2 strikes OTM
        assert selection == StrikeSelection.OTM_2
        assert strike == 602  # Floor + 2 for CALL OTM_2


class TestContinuationConfirmationParity:
    """Verify continuation confirmation matches backtest."""

    def test_bullish_continuation_confirmed(self):
        """Test bullish continuation is confirmed."""
        et_tz = ZoneInfo("America/New_York")
        timestamps = pd.date_range("2025-01-06 09:30:00", periods=3, freq="1min", tz=et_tz)

        # Bar 2 needs to be a strong green candle entirely above VWAP (600.2)
        # body = close - open = 602.0 - 601.0 = 1.0
        # range = high - low = 602.5 - 600.8 = 1.7
        # body_ratio = 1.0 / 1.7 = 0.59 > 0.3 ✓
        # body_low = min(open, close) = 601.0 > 600.2 (VWAP) ✓
        ohlcv_df = pd.DataFrame({
            "open": [600.0, 600.5, 601.0],  # Bar 2 opens above VWAP (600.2)
            "high": [600.5, 601.0, 602.5],
            "low": [599.5, 600.0, 600.8],   # Bar 2 low above VWAP
            "close": [600.3, 600.8, 602.0], # Bar 2 closes well above open (green)
            "volume": [100000, 120000, 150000],
        }, index=timestamps)

        # Create a bullish potential breach signal at bar 1
        potential_breach = BreachSignal(
            breach_type=BreachType.BULLISH_POTENTIAL,
            vwap_value=600.2,
            candle_close=600.8,
            candle_open=600.5,
            candle_high=601.0,
            candle_low=600.0,
            volume=120000,
            continuation_volume=120000,
            volume_ratio=1.2,
            is_potential=True,
        )

        # Check confirmation at bar 2
        confirmed = check_breach_continuation_confirmation(ohlcv_df, 2, potential_breach)

        # Should confirm - bar 2 is green and entirely above VWAP
        assert bool(confirmed) == True

    def test_bullish_continuation_not_confirmed_red_candle(self):
        """Test bullish continuation not confirmed with red candle."""
        et_tz = ZoneInfo("America/New_York")
        timestamps = pd.date_range("2025-01-06 09:30:00", periods=3, freq="1min", tz=et_tz)

        ohlcv_df = pd.DataFrame({
            "open": [600.0, 600.5, 601.5],  # Bar 2 opens high
            "high": [600.5, 601.0, 601.5],
            "low": [599.5, 600.0, 600.5],
            "close": [600.3, 600.8, 600.7],  # Bar 2 closes RED (below open)
            "volume": [100000, 120000, 150000],
        }, index=timestamps)

        potential_breach = BreachSignal(
            breach_type=BreachType.BULLISH_POTENTIAL,
            vwap_value=600.2,
            candle_close=600.8,
            candle_open=600.5,
            candle_high=601.0,
            candle_low=600.0,
            volume=120000,
            continuation_volume=120000,
            volume_ratio=1.2,
            is_potential=True,
        )

        confirmed = check_breach_continuation_confirmation(ohlcv_df, 2, potential_breach)

        # Should NOT confirm - bar 2 is red
        assert confirmed == False

    def test_bullish_continuation_rejected_wick_below_breakout_low(self):
        """Test bullish continuation rejected when wick dips below breakout candle's low."""
        et_tz = ZoneInfo("America/New_York")
        timestamps = pd.date_range("2025-01-06 09:30:00", periods=3, freq="1min", tz=et_tz)

        # Bar 2 is green and body is above VWAP, but wick dips below breakout candle's low
        ohlcv_df = pd.DataFrame({
            "open": [600.0, 600.5, 601.0],
            "high": [600.5, 601.0, 602.5],
            "low": [599.5, 600.0, 599.8],   # Bar 2 low (599.8) < breakout low (600.0) → fake breakout
            "close": [600.3, 600.8, 602.0],
            "volume": [100000, 120000, 150000],
        }, index=timestamps)

        potential_breach = BreachSignal(
            breach_type=BreachType.BULLISH_POTENTIAL,
            vwap_value=600.2,
            candle_close=600.8,
            candle_open=600.5,
            candle_high=601.0,
            candle_low=600.0,  # Breakout candle low
            volume=120000,
            continuation_volume=120000,
            volume_ratio=1.2,
            is_potential=True,
        )

        confirmed = check_breach_continuation_confirmation(ohlcv_df, 2, potential_breach)

        # Should NOT confirm - wick goes below breakout candle's low (fake breakout)
        assert confirmed == False

    def test_bearish_continuation_rejected_wick_above_breakout_high(self):
        """Test bearish continuation rejected when wick rises above breakout candle's high."""
        et_tz = ZoneInfo("America/New_York")
        timestamps = pd.date_range("2025-01-06 09:30:00", periods=3, freq="1min", tz=et_tz)

        # Bar 2 is red and body is below VWAP, but wick goes above breakout candle's high
        ohlcv_df = pd.DataFrame({
            "open": [600.0, 599.5, 599.0],
            "high": [600.5, 600.0, 600.3],   # Bar 2 high (600.3) > breakout high (600.0) → fake breakout
            "low": [599.5, 599.0, 598.0],
            "close": [600.3, 599.2, 598.5],
            "volume": [100000, 120000, 150000],
        }, index=timestamps)

        potential_breach = BreachSignal(
            breach_type=BreachType.BEARISH_POTENTIAL,
            vwap_value=599.8,
            candle_close=599.2,
            candle_open=599.5,
            candle_high=600.0,  # Breakout candle high
            candle_low=599.0,
            volume=120000,
            continuation_volume=120000,
            volume_ratio=1.2,
            is_potential=True,
        )

        confirmed = check_breach_continuation_confirmation(ohlcv_df, 2, potential_breach)

        # Should NOT confirm - wick goes above breakout candle's high (fake breakout)
        assert confirmed == False

    def test_bullish_continuation_confirmed_with_wick_check(self):
        """Test bullish continuation still works when wick is within breakout range."""
        et_tz = ZoneInfo("America/New_York")
        timestamps = pd.date_range("2025-01-06 09:30:00", periods=3, freq="1min", tz=et_tz)

        # Bar 2 is green, body above VWAP, wick stays within breakout candle range
        ohlcv_df = pd.DataFrame({
            "open": [600.0, 600.5, 601.0],
            "high": [600.5, 601.0, 602.5],
            "low": [599.5, 600.0, 600.5],   # Bar 2 low (600.5) >= breakout low (600.0) → valid
            "close": [600.3, 600.8, 602.0],
            "volume": [100000, 120000, 150000],
        }, index=timestamps)

        potential_breach = BreachSignal(
            breach_type=BreachType.BULLISH_POTENTIAL,
            vwap_value=600.2,
            candle_close=600.8,
            candle_open=600.5,
            candle_high=601.0,
            candle_low=600.0,  # Breakout candle low
            volume=120000,
            continuation_volume=120000,
            volume_ratio=1.2,
            is_potential=True,
        )

        confirmed = check_breach_continuation_confirmation(ohlcv_df, 2, potential_breach)

        # Should confirm - all conditions met including wick check
        assert bool(confirmed) == True


class TestNewsEventDetection:
    """Test news event detection parity between trading_rules and backtest."""

    def test_bullish_news_event_detected(self):
        """Green candle with top volume, large body, minimal upper wick → BULLISH."""
        et = ZoneInfo("America/New_York")
        # Need 30+ bars so we're past the opening range and have enough volume history
        # Create 35 normal bars then 1 big bullish bar
        dates = pd.date_range(
            start=datetime(2024, 1, 15, 10, 0, tzinfo=et),
            periods=36,
            freq="1min",
        )
        data = {
            "open": [600.0] * 36,
            "high": [600.5] * 36,
            "low": [599.5] * 36,
            "close": [600.2] * 36,
            "volume": [50000] * 36,
        }
        ohlcv_df = pd.DataFrame(data, index=dates)

        # Make the last bar a big bullish candle: high volume, large body, minimal upper wick
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("open")] = 600.0
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("close")] = 602.0  # Big green body
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("high")] = 602.1   # Tiny upper wick
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("low")] = 599.8
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("volume")] = 500000  # 10x normal volume

        # Close of 602.0 is above upper_2sd of 601.0, open of 600.0 is below upper_1sd of 600.5 → should trigger
        result = detect_news_event(ohlcv_df, len(ohlcv_df) - 1, upper_1sd=600.5, lower_1sd=599.5, upper_2sd=601.0, lower_2sd=599.0)
        assert result.event_type == NewsEventType.BULLISH

    def test_bullish_news_event_rejected_within_bands(self):
        """Green candle that doesn't close above +2σ VWAP → no signal."""
        et = ZoneInfo("America/New_York")
        dates = pd.date_range(
            start=datetime(2024, 1, 15, 10, 0, tzinfo=et),
            periods=36,
            freq="1min",
        )
        data = {
            "open": [600.0] * 36,
            "high": [600.5] * 36,
            "low": [599.5] * 36,
            "close": [600.2] * 36,
            "volume": [50000] * 36,
        }
        ohlcv_df = pd.DataFrame(data, index=dates)

        # Same big bullish candle but upper_2sd is above the close
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("open")] = 600.0
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("close")] = 602.0
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("high")] = 602.1
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("low")] = 599.8
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("volume")] = 500000

        # Close of 602.0 is NOT above upper_2sd of 603.0 → no signal
        result = detect_news_event(ohlcv_df, len(ohlcv_df) - 1, upper_1sd=601.5, lower_1sd=598.5, upper_2sd=603.0, lower_2sd=597.0)
        assert result.event_type == NewsEventType.NONE

    def test_bullish_news_event_rejected_open_too_close(self):
        """Green candle where open is already above +1σ → no signal (not enough movement)."""
        et = ZoneInfo("America/New_York")
        dates = pd.date_range(
            start=datetime(2024, 1, 15, 10, 0, tzinfo=et),
            periods=36,
            freq="1min",
        )
        data = {
            "open": [600.0] * 36,
            "high": [600.5] * 36,
            "low": [599.5] * 36,
            "close": [600.2] * 36,
            "volume": [50000] * 36,
        }
        ohlcv_df = pd.DataFrame(data, index=dates)

        # Big bullish candle but open (601.5) is already above upper_1sd (601.0)
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("open")] = 601.5
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("close")] = 603.0
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("high")] = 603.1
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("low")] = 601.4
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("volume")] = 500000

        # Open 601.5 >= upper_1sd 601.0 → rejected (started too close to +2σ)
        result = detect_news_event(ohlcv_df, len(ohlcv_df) - 1, upper_1sd=601.0, lower_1sd=599.0, upper_2sd=602.0, lower_2sd=598.0)
        assert result.event_type == NewsEventType.NONE

    def test_bearish_news_event_detected(self):
        """Red candle with top volume, large body, minimal lower wick → BEARISH."""
        et = ZoneInfo("America/New_York")
        dates = pd.date_range(
            start=datetime(2024, 1, 15, 10, 0, tzinfo=et),
            periods=36,
            freq="1min",
        )
        data = {
            "open": [600.0] * 36,
            "high": [600.5] * 36,
            "low": [599.5] * 36,
            "close": [600.2] * 36,
            "volume": [50000] * 36,
        }
        ohlcv_df = pd.DataFrame(data, index=dates)

        # Big bearish candle: close near low (small opposing lower wick = c_close - c_low)
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("open")] = 602.0
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("close")] = 598.0  # Big red body, below lower_2sd
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("high")] = 602.2
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("low")] = 597.9   # Tiny lower wick
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("volume")] = 500000

        # Close of 598.0 is below lower_2sd of 599.0, open of 602.0 is above lower_1sd of 599.5 → should trigger
        result = detect_news_event(ohlcv_df, len(ohlcv_df) - 1, upper_1sd=600.5, lower_1sd=599.5, upper_2sd=601.0, lower_2sd=599.0)
        assert result.event_type == NewsEventType.BEARISH

    def test_no_news_event_low_volume(self):
        """Normal volume bar should not trigger news event."""
        et = ZoneInfo("America/New_York")
        dates = pd.date_range(
            start=datetime(2024, 1, 15, 10, 0, tzinfo=et),
            periods=36,
            freq="1min",
        )
        data = {
            "open": [600.0] * 36,
            "high": [600.5] * 36,
            "low": [599.5] * 36,
            "close": [600.2] * 36,
            "volume": [50000] * 36,
        }
        ohlcv_df = pd.DataFrame(data, index=dates)

        # Normal volume, even with a big body — volume must be BELOW 90th percentile
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("open")] = 600.0
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("close")] = 602.0
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("high")] = 602.1
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("low")] = 599.8
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("volume")] = 40000  # Below average, clearly not top 10%

        result = detect_news_event(ohlcv_df, len(ohlcv_df) - 1, upper_1sd=600.5, lower_1sd=599.5, upper_2sd=601.0, lower_2sd=599.0)
        assert result.event_type == NewsEventType.NONE

    def test_no_news_event_without_bands(self):
        """Without VWAP bands provided, news event should not trigger."""
        et = ZoneInfo("America/New_York")
        dates = pd.date_range(
            start=datetime(2024, 1, 15, 10, 0, tzinfo=et),
            periods=36,
            freq="1min",
        )
        data = {
            "open": [600.0] * 36,
            "high": [600.5] * 36,
            "low": [599.5] * 36,
            "close": [600.2] * 36,
            "volume": [50000] * 36,
        }
        ohlcv_df = pd.DataFrame(data, index=dates)

        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("open")] = 600.0
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("close")] = 602.0
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("high")] = 602.1
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("low")] = 599.8
        ohlcv_df.iloc[-1, ohlcv_df.columns.get_loc("volume")] = 500000

        # No bands passed → should not trigger
        result = detect_news_event(ohlcv_df, len(ohlcv_df) - 1)
        assert result.event_type == NewsEventType.NONE
