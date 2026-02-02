"""
Position sizing and strike selection logic.

Contains rules for calculating position sizes and selecting option strikes
based on signal type, confidence, and market conditions.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

from src.backtest.types import PositionType


class StrikeSelection(Enum):
    """Strike selection mode."""
    ITM = "itm"  # In the money
    ATM = "atm"  # At the money
    OTM = "otm"  # Out of the money
    OTM_2 = "otm_2"  # 2 strikes OTM


@dataclass
class PositionSizeConfig:
    """Configuration for position sizing."""
    # Base position sizes (as percentage of capital)
    base_position_pct: float = 0.03  # 3% default
    breach_position_pct: float = 0.03  # 3% for VWAP breach trades
    reversal_position_pct: float = 0.03  # 3% for reversal trades
    reversal_engulfing_position_pct: float = 0.05  # 5% for engulfing reversal trades
    news_event_position_pct: float = 0.05  # 5% for news event trades
    max_position_pct: float = 0.05  # 5% maximum

    # Power hour boost
    power_hour_start_minutes: int = 14 * 60  # 2:00 PM ET (minutes from midnight)
    power_hour_end_minutes: int = 16 * 60  # 4:00 PM ET (minutes from midnight)
    power_hour_bonus_pct: float = 0.02  # Additional 2% during power hour

    # Capital-based multiplier
    low_capital_threshold: float = 15000.0
    low_capital_multiplier: float = 2.0  # 2x position size when under threshold

    # Strike selection time boundaries (minutes from midnight)
    otm_time_before: int = 10 * 60 + 30  # Before 10:30 AM ET
    otm_time_after: int = 15 * 60  # After 3:00 PM ET


@dataclass
class PositionSizeResult:
    """Result of position size calculation."""
    num_contracts: int
    position_value: float
    strike: float
    strike_selection: StrikeSelection
    size_multiplier: float
    position_pct: float


class PositionSizer:
    """
    Calculates position sizes and selects option strikes.

    Position sizing rules:
    - Breach trades: 3% of capital + bonus
    - Normal trades: 3% of capital + bonus
    - Power hour (2-4pm): Additional 2% bonus
    - Low capital (<$15k): 2x multiplier

    Strike selection rules:
    - Breach trades: ATM
    - Early trades (9:45-10:00): ATM
    - Normal trades: Based on confidence (ITM default, OTM for high confidence)
    """

    def __init__(self, config: Optional[PositionSizeConfig] = None):
        """
        Initialize the position sizer.

        Args:
            config: Position sizing configuration
        """
        self.config = config or PositionSizeConfig()
        self.et_tz = ZoneInfo("America/New_York")

    def calculate_position_size(
        self,
        capital: float,
        option_price: float,
        signal_type: str,  # "breach", "early", "normal"
        time_of_day: datetime,
        position_size_bonus: float = 0.0,
    ) -> PositionSizeResult:
        """
        Calculate position size in contracts and dollar value.

        Args:
            capital: Current account capital
            option_price: Price per option contract (per share)
            signal_type: Type of signal triggering the trade
            time_of_day: Current time
            position_size_bonus: Additional bonus from heuristic confluence (0.0 to 0.02)

        Returns:
            PositionSizeResult with contract count, value, and sizing details
        """
        # Get time in minutes from midnight (ET)
        et_time = time_of_day.astimezone(self.et_tz) if time_of_day.tzinfo else time_of_day
        time_minutes = et_time.hour * 60 + et_time.minute

        # Capital-based multiplier
        size_multiplier = (
            self.config.low_capital_multiplier
            if capital < self.config.low_capital_threshold
            else 1.0
        )

        # Power hour bonus (overrides heuristic bonus)
        if self.config.power_hour_start_minutes <= time_minutes < self.config.power_hour_end_minutes:
            position_size_bonus = self.config.power_hour_bonus_pct

        # Base position percentage based on signal type
        if signal_type == "news_event":
            base_pct = self.config.news_event_position_pct
            total_pct = min(base_pct + position_size_bonus, self.config.max_position_pct)
        elif signal_type == "breach":
            base_pct = self.config.breach_position_pct
            total_pct = min(base_pct + position_size_bonus, self.config.max_position_pct)
        elif signal_type == "double_breakout":
            base_pct = self.config.breach_position_pct
            total_pct = min(base_pct + position_size_bonus, self.config.max_position_pct)
        elif signal_type == "reversal_engulfing":
            base_pct = self.config.reversal_engulfing_position_pct
            total_pct = min(base_pct + position_size_bonus, self.config.max_position_pct)
        elif signal_type == "reversal":
            base_pct = self.config.reversal_position_pct
            total_pct = min(base_pct + position_size_bonus, self.config.max_position_pct)
        elif signal_type == "bounce_engulfing":
            base_pct = self.config.reversal_engulfing_position_pct  # 5%
            total_pct = min(base_pct + position_size_bonus, self.config.max_position_pct)
        elif signal_type == "bounce":
            base_pct = self.config.reversal_position_pct  # 3%
            total_pct = min(base_pct + position_size_bonus, self.config.max_position_pct)
        else:
            base_pct = self.config.base_position_pct
            total_pct = min(base_pct + position_size_bonus, self.config.max_position_pct)

        # Calculate target position value
        target_position_value = capital * total_pct * size_multiplier

        # Calculate number of contracts
        cost_per_contract = option_price * 100
        if cost_per_contract <= 0:
            return PositionSizeResult(
                num_contracts=0,
                position_value=0,
                strike=0,
                strike_selection=StrikeSelection.ATM,
                size_multiplier=size_multiplier,
                position_pct=total_pct,
            )

        num_contracts = max(1, int(target_position_value / cost_per_contract))
        actual_position_value = num_contracts * cost_per_contract

        return PositionSizeResult(
            num_contracts=num_contracts,
            position_value=actual_position_value,
            strike=0,  # To be filled by select_strike()
            strike_selection=StrikeSelection.ATM,  # To be filled by select_strike()
            size_multiplier=size_multiplier,
            position_pct=total_pct,
        )

    def select_strike(
        self,
        underlying_price: float,
        position_type: PositionType,
        signal_type: str,
        confidence: float,
        time_of_day: datetime,
    ) -> Tuple[float, StrikeSelection]:
        """
        Select strike price based on signal type, confidence, and time of day.

        Args:
            underlying_price: Current underlying (SPY) price
            position_type: CALL or PUT
            signal_type: Type of signal ("breach", "early", "normal")
            confidence: Model confidence (0.0 to 1.0)
            time_of_day: Current time

        Returns:
            Tuple of (strike_price, strike_selection_type)
        """
        # Get time in minutes from midnight (ET)
        et_time = time_of_day.astimezone(self.et_tz) if time_of_day.tzinfo else time_of_day
        time_minutes = et_time.hour * 60 + et_time.minute
        is_otm_time = (
            time_minutes < self.config.otm_time_before or
            time_minutes >= self.config.otm_time_after
        )

        # News event trades: Always ATM
        if signal_type == "news_event":
            return self._select_atm_strike(underlying_price)

        # Breach trades: Always ATM
        if signal_type == "breach":
            return self._select_atm_strike(underlying_price)

        # Double breakout: Always ATM
        if signal_type == "double_breakout":
            return self._select_atm_strike(underlying_price)

        # Strong reversal (engulfing): OTM
        if signal_type == "reversal_engulfing":
            return self._select_otm_strike(underlying_price, position_type, levels=1)

        # Normal reversal: ATM
        if signal_type == "reversal":
            return self._select_atm_strike(underlying_price)

        # Strong bounce (engulfing): OTM
        if signal_type == "bounce_engulfing":
            return self._select_otm_strike(underlying_price, position_type, levels=1)

        # Normal bounce: ATM
        if signal_type == "bounce":
            return self._select_atm_strike(underlying_price)

        # Early trades: Always ATM
        if signal_type == "early":
            return self._select_atm_strike(underlying_price)

        # Normal trades: Confidence-based selection
        if confidence >= 0.90:
            return self._select_otm_strike(underlying_price, position_type, levels=2)
        elif confidence >= 0.60:
            return self._select_otm_strike(underlying_price, position_type, levels=1)
        else:
            return self._select_itm_strike(underlying_price, position_type)

    def _select_itm_strike(
        self,
        underlying_price: float,
        position_type: PositionType,
    ) -> Tuple[float, StrikeSelection]:
        """Select ITM strike."""
        if position_type == PositionType.CALL:
            strike = int(underlying_price)  # ITM for calls (floor)
        else:
            strike = int(underlying_price) + 1  # ITM for puts (ceiling)
        return strike, StrikeSelection.ITM

    def _select_atm_strike(
        self,
        underlying_price: float,
    ) -> Tuple[float, StrikeSelection]:
        """Select ATM strike."""
        strike = round(underlying_price)
        return strike, StrikeSelection.ATM

    def _select_otm_strike(
        self,
        underlying_price: float,
        position_type: PositionType,
        levels: int = 1,
    ) -> Tuple[float, StrikeSelection]:
        """Select OTM strike."""
        if position_type == PositionType.CALL:
            strike = int(underlying_price) + levels
        else:
            strike = int(underlying_price) + 1 - levels

        selection = StrikeSelection.OTM if levels == 1 else StrikeSelection.OTM_2
        return strike, selection

    def calculate_position_with_strike(
        self,
        capital: float,
        option_price: float,
        underlying_price: float,
        position_type: PositionType,
        signal_type: str,
        confidence: float,
        time_of_day: datetime,
        position_size_bonus: float = 0.0,
    ) -> PositionSizeResult:
        """
        Calculate position size and select strike in one call.

        This is a convenience method that combines calculate_position_size()
        and select_strike().

        Args:
            capital: Current account capital
            option_price: Price per option contract (per share)
            underlying_price: Current underlying (SPY) price
            position_type: CALL or PUT
            signal_type: Type of signal
            confidence: Model confidence
            time_of_day: Current time
            position_size_bonus: Additional bonus from heuristic confluence

        Returns:
            PositionSizeResult with all fields populated
        """
        # First select strike
        strike, strike_selection = self.select_strike(
            underlying_price, position_type, signal_type, confidence, time_of_day
        )

        # Then calculate position size
        result = self.calculate_position_size(
            capital, option_price, signal_type, time_of_day, position_size_bonus
        )

        # Update result with strike info
        result.strike = strike
        result.strike_selection = strike_selection

        return result
