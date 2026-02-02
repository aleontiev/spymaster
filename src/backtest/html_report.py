"""
HTML Report Generator for Multi-Scale Backtest.

Generates interactive TradingView-style charts with:
- OHLCV candlestick data with volume subchart
- Trade entries/exits with win/loss coloring
- Individual model signals (1m, 5m, 15m) as toggleable series
- Daily boundary lines with date labels
- Bidirectional trade selection (table <-> chart)
- Multi-timeframe candles (1m, 15m, 1h, 1d, 1w) with auto-switching
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def aggregate_ohlcv(ohlcv_df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Aggregate 1-minute OHLCV data to a larger timeframe.

    Args:
        ohlcv_df: DataFrame with DatetimeIndex and columns: open, high, low, close, volume
        timeframe: One of '15m', '1h', '1d', '1w'

    Returns:
        Aggregated DataFrame
    """
    # Map timeframe to pandas resample rule
    resample_map = {
        '5m': '5min',
        '15m': '15min',
        '1h': '1h',
        '1d': '1D',
        '1w': '1W',
    }

    if timeframe not in resample_map:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    rule = resample_map[timeframe]

    # Resample using OHLCV aggregation rules
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }
    if 'volume' in ohlcv_df.columns:
        agg_dict['volume'] = 'sum'

    resampled = ohlcv_df.resample(rule).agg(agg_dict)

    # Drop rows where we have no data (NaN from empty periods)
    resampled = resampled.dropna(subset=['open', 'close'])

    return resampled


def calculate_vwap_with_bands(ohlcv_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Calculate VWAP with standard deviation bands anchored at start of each trading day.

    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    Standard deviation bands at +-1 and +-2 standard deviations.
    Resets at start of each new trading day.

    Returns:
        Tuple of (vwap, upper_1sd, lower_1sd, upper_2sd, lower_2sd) Series
    """
    # Typical price = (high + low + close) / 3
    typical_price = (ohlcv_df['high'] + ohlcv_df['low'] + ohlcv_df['close']) / 3
    volume = ohlcv_df.get('volume', pd.Series(1, index=ohlcv_df.index))

    vwap_values = []
    upper_1sd_values = []
    lower_1sd_values = []
    upper_2sd_values = []
    lower_2sd_values = []

    current_date = None
    cum_tp_vol = 0.0
    cum_vol = 0.0
    cum_tp_sq_vol = 0.0  # For variance calculation

    for i, (idx, row) in enumerate(ohlcv_df.iterrows()):
        dt = idx.tz_convert('America/New_York') if idx.tzinfo else idx
        row_date = dt.date()

        # Reset at start of new day
        if row_date != current_date:
            current_date = row_date
            cum_tp_vol = 0.0
            cum_vol = 0.0
            cum_tp_sq_vol = 0.0

        tp = typical_price.iloc[i] if not np.isnan(typical_price.iloc[i]) else 0
        vol = volume.iloc[i] if not np.isnan(volume.iloc[i]) else 0

        cum_tp_vol += tp * vol
        cum_vol += vol
        cum_tp_sq_vol += (tp ** 2) * vol

        if cum_vol > 0:
            vwap = cum_tp_vol / cum_vol
            # Variance = E[X^2] - E[X]^2 (volume-weighted)
            variance = (cum_tp_sq_vol / cum_vol) - (vwap ** 2)
            std_dev = np.sqrt(max(0, variance))  # Ensure non-negative

            vwap_values.append(vwap)
            upper_1sd_values.append(vwap + std_dev)
            lower_1sd_values.append(vwap - std_dev)
            upper_2sd_values.append(vwap + 2 * std_dev)
            lower_2sd_values.append(vwap - 2 * std_dev)
        else:
            # Fallback to typical price if no volume
            tp_val = typical_price.iloc[i]
            vwap_values.append(tp_val)
            upper_1sd_values.append(tp_val)
            lower_1sd_values.append(tp_val)
            upper_2sd_values.append(tp_val)
            lower_2sd_values.append(tp_val)

    return (
        pd.Series(vwap_values, index=ohlcv_df.index),
        pd.Series(upper_1sd_values, index=ohlcv_df.index),
        pd.Series(lower_1sd_values, index=ohlcv_df.index),
        pd.Series(upper_2sd_values, index=ohlcv_df.index),
        pd.Series(lower_2sd_values, index=ohlcv_df.index),
    )


def calculate_vwap(ohlcv_df: pd.DataFrame) -> pd.Series:
    """
    Calculate VWAP anchored at start of each trading day.

    VWAP = cumulative(typical_price * volume) / cumulative(volume)
    Resets at start of each new trading day.
    """
    vwap, _, _, _, _ = calculate_vwap_with_bands(ohlcv_df)
    return vwap


def ohlcv_to_json(ohlcv_df: pd.DataFrame, include_vwap: bool = False) -> Tuple[List[Dict], List[Dict], Dict[str, List[Dict]]]:
    """Convert OHLCV DataFrame to JSON format for Lightweight Charts.

    Returns:
        Tuple of (ohlcv_data, volume_data, vwap_dict) where vwap_dict contains:
        - 'vwap': main VWAP line
        - 'upper_1sd': +1 standard deviation band
        - 'lower_1sd': -1 standard deviation band
        - 'upper_2sd': +2 standard deviation band
        - 'lower_2sd': -2 standard deviation band
    """
    ohlcv_data = []
    volume_data = []
    vwap_dict = {
        'vwap': [],
        'upper_1sd': [],
        'lower_1sd': [],
        'upper_2sd': [],
        'lower_2sd': [],
    }

    # Calculate VWAP with bands if requested
    if include_vwap:
        vwap_series, upper_1sd, lower_1sd, upper_2sd, lower_2sd = calculate_vwap_with_bands(ohlcv_df)

    for i, (idx, row) in enumerate(ohlcv_df.iterrows()):
        ts = int(idx.timestamp())
        ohlcv_data.append({
            "time": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        })
        # Volume with color based on candle direction
        is_up = row["close"] >= row["open"]
        vol = float(row.get("volume", 0)) if "volume" in row.index and not np.isnan(row.get("volume", 0)) else 0
        volume_data.append({
            "time": ts,
            "value": vol,
            "color": "rgba(38, 166, 154, 0.5)" if is_up else "rgba(239, 83, 80, 0.5)",
        })

        if include_vwap:
            vwap_val = vwap_series.iloc[i]
            if not np.isnan(vwap_val):
                vwap_dict['vwap'].append({"time": ts, "value": float(vwap_val)})
                vwap_dict['upper_1sd'].append({"time": ts, "value": float(upper_1sd.iloc[i])})
                vwap_dict['lower_1sd'].append({"time": ts, "value": float(lower_1sd.iloc[i])})
                vwap_dict['upper_2sd'].append({"time": ts, "value": float(upper_2sd.iloc[i])})
                vwap_dict['lower_2sd'].append({"time": ts, "value": float(lower_2sd.iloc[i])})

    return ohlcv_data, volume_data, vwap_dict


@dataclass
class SignalMarker:
    """A signal marker for the chart."""
    timestamp: int  # Unix timestamp in seconds
    signal_type: str  # "long" or "short"
    model: str  # "1m", "5m", "15m"
    price: float  # Price level for marker placement


@dataclass
class TradeMarker:
    """A trade marker for the chart."""
    entry_time: int  # Unix timestamp
    exit_time: int  # Unix timestamp
    entry_price: float
    exit_price: float
    is_long: bool
    is_win: bool
    pnl_pct: float
    pnl_dollars: float
    agreeing_models: Tuple[str, ...]
    exit_reason: str
    capital_after: float = 0.0  # Balance after trade
    contract: str = ""  # e.g., "SPY 600C"
    entry_option_price: float = 0.0  # Cost basis per contract
    exit_option_price: float = 0.0  # Exit price per contract
    num_contracts: int = 10  # Number of contracts
    is_runner: bool = False  # True if this is a runner continuation
    parent_entry_time: int = 0  # Original entry time for runners (for matching entry markers)
    dominant_model: str = ""  # Signal type: "breach", "1m", "5m", "15m"
    quality: str = ""  # Trade quality grade: A, B, C, D


def generate_html_report(
    ohlcv_df: pd.DataFrame,
    trades: List[TradeMarker],
    signals: Dict[str, List[SignalMarker]],  # {"1m": [...], "5m": [...], "15m": [...]}
    combination_name: str,
    output_path: Path,
    title: str = "Multi-Scale Backtest Report",
    command: Optional[str] = None,
    missed_signals: Optional[List[Dict]] = None,
    metrics: Optional[Dict] = None,
) -> Path:
    """
    Generate an HTML report with interactive TradingView chart.

    Args:
        ohlcv_df: DataFrame with columns: timestamp (index), open, high, low, close, volume
        trades: List of TradeMarker objects
        signals: Dict mapping model name to list of SignalMarker objects
        combination_name: Name of the model combination (e.g., "1m+15m")
        output_path: Path to save the HTML file
        title: Report title
        command: Command used to generate the report
        missed_signals: List of missed triple-confluence signals with rejection reasons
        metrics: Optional dict with additional metrics (max_daily_drawdown_pct, max_weekly_drawdown_pct, etc.)

    Returns:
        Path to the generated HTML file
    """
    # Default metrics if not provided
    if metrics is None:
        metrics = {}

    # Convert OHLCV to JSON format for Lightweight Charts - all timeframes
    # 1-minute data (original) - include VWAP with bands for the chart
    ohlcv_1m, volume_1m, vwap_data = ohlcv_to_json(ohlcv_df, include_vwap=True)

    # Create VWAP lookup dictionary (timestamp -> vwap value) for trade analysis
    vwap_lookup = {v["time"]: v["value"] for v in vwap_data['vwap']}

    # Aggregate to higher timeframes (no VWAP for aggregated - we use 1m VWAP on chart)
    ohlcv_5m_df = aggregate_ohlcv(ohlcv_df, '5m')
    ohlcv_15m_df = aggregate_ohlcv(ohlcv_df, '15m')
    ohlcv_1h_df = aggregate_ohlcv(ohlcv_df, '1h')
    ohlcv_1d_df = aggregate_ohlcv(ohlcv_df, '1d')
    ohlcv_1w_df = aggregate_ohlcv(ohlcv_df, '1w')

    ohlcv_5m, volume_5m, _ = ohlcv_to_json(ohlcv_5m_df)
    ohlcv_15m, volume_15m, _ = ohlcv_to_json(ohlcv_15m_df)
    ohlcv_1h, volume_1h, _ = ohlcv_to_json(ohlcv_1h_df)
    ohlcv_1d, volume_1d, _ = ohlcv_to_json(ohlcv_1d_df)
    ohlcv_1w, volume_1w, _ = ohlcv_to_json(ohlcv_1w_df)

    # Keep ohlcv_data as 1m for backwards compatibility with rest of code
    ohlcv_data = ohlcv_1m
    volume_data = volume_1m

    # Calculate daily boundaries - find the FIRST candle of each new day for vertical line placement
    daily_boundaries = []
    prev_date = None
    for idx, row in ohlcv_df.iterrows():
        ts = int(idx.timestamp())
        dt = idx.tz_convert("America/New_York") if idx.tzinfo else idx
        current_date = dt.date()

        # When we see a new date, mark the first candle of the new day
        if prev_date is not None and current_date != prev_date:
            date_str = current_date.strftime("%b %d")
            daily_boundaries.append({
                "time": ts,
                "label": date_str,
            })

        prev_date = current_date

    # Convert signals to JSON - use arrowUp/arrowDown for triangular markers
    signals_json = {}
    for model, signal_list in signals.items():
        signals_json[model] = [
            {
                "time": s.timestamp,
                "position": "aboveBar" if s.signal_type == "short" else "belowBar",
                "color": "#ef5350" if s.signal_type == "short" else "#26a69a",
                "shape": "arrowDown" if s.signal_type == "short" else "arrowUp",
                "text": "",
                "size": 0.5,
            }
            for s in signal_list
        ]

    # Convert trades to entry/exit markers
    # Track which entry times we've already added markers for (to avoid duplicates from runners)
    seen_entry_times = set()
    trade_entries = []
    trade_exits = []

    for i, t in enumerate(trades):
        # Determine the actual entry time for this trade's marker
        entry_ts = t.parent_entry_time if t.is_runner and t.parent_entry_time else t.entry_time

        # Only add entry marker if we haven't added one for this timestamp yet
        # This consolidates runner entries with their parent
        if entry_ts not in seen_entry_times:
            seen_entry_times.add(entry_ts)
            entry_color = "#26a69a" if t.is_long else "#ef5350"
            trade_entries.append({
                "time": entry_ts,
                "position": "belowBar" if t.is_long else "aboveBar",
                "color": entry_color,
                "shape": "arrowUp" if t.is_long else "arrowDown",
                "text": "",
                "size": 2,
                "tradeIdx": i,
                "isLong": t.is_long,
            })

        # Exit marker - always add at the ACTUAL exit time for this trade
        # (runners have their own exit time different from parent)
        exit_color = "#26a69a" if t.is_win else "#ef5350"
        exit_label = f"{t.pnl_pct:+.0f}%"
        trade_exits.append({
            "time": t.exit_time,  # Use actual exit time, not parent's
            "position": "aboveBar" if t.is_long else "belowBar",
            "color": exit_color,
            "shape": "circle",
            "text": exit_label,
            "size": 2,
            "tradeIdx": i,
            "isRunner": t.is_runner,
        })

    # Generate trades table HTML with click-to-navigate functionality
    trades_table_rows = []
    for i, t in enumerate(trades):
        entry_dt = datetime.fromtimestamp(t.entry_time).strftime("%Y-%m-%d %H:%M")
        # Calculate duration in minutes
        duration_mins = int((t.exit_time - t.entry_time) / 60)
        duration_str = f"{duration_mins}m"
        result_class = "win" if t.is_win else "loss"
        if t.is_runner:
            result_class += " runner"
        contract = t.contract if t.contract else ("CALL" if t.is_long else "PUT")
        contract_color = "#26a69a" if t.is_long else "#ef5350"  # Green for long, red for short

        # Derive entry reason from dominant_model
        if t.dominant_model == "breach":
            entry_reason = "Breakout"
        elif t.dominant_model == "bounce":
            entry_reason = "Bounce"
        elif t.dominant_model == "news_event":
            entry_reason = "News Event"
        elif t.dominant_model in ("1m", "5m", "15m"):
            entry_reason = "Continuation"
        elif t.dominant_model:
            entry_reason = t.dominant_model
        else:
            entry_reason = "Unknown"

        trades_table_rows.append(f"""
            <tr class="{result_class}" data-trade-idx="{i}" data-entry-time="{t.entry_time}" data-exit-time="{t.exit_time}" onclick="focusOnTrade({i}, {t.entry_time}, {t.exit_time})" style="cursor: pointer;">
                <td>{i+1}</td>
                <td>{entry_dt}</td>
                <td>{duration_str}</td>
                <td style="color: {contract_color}; font-weight: bold;">{contract}</td>
                <td>{t.num_contracts}</td>
                <td>${t.entry_option_price:.2f}</td>
                <td>${t.exit_option_price:.2f}</td>
                <td class="pnl">{t.pnl_pct:+.2f}%</td>
                <td class="pnl">${t.pnl_dollars:+,.2f}</td>
                <td>${t.capital_after:,.0f}</td>
                <td style="font-weight: bold; color: {'#26a69a' if t.quality == 'A' else '#66bb6a' if t.quality == 'B' else '#ffa726' if t.quality == 'C' else '#ef5350' if t.quality == 'D' else '#888'};">{t.quality}</td>
                <td>{entry_reason}</td>
                <td>{t.exit_reason}</td>
            </tr>
        """)

    trades_table_html = "\n".join(trades_table_rows)

    # Calculate summary stats
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.is_win)
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    total_pnl = sum(t.pnl_dollars for t in trades)

    # Calculate best/worst alltime P/L (max gain and max drawdown from initial capital)
    initial_capital = trades[0].capital_after - trades[0].pnl_dollars if trades else 25000
    running_capital = initial_capital
    peak_capital = initial_capital
    lowest_capital = initial_capital
    max_gain_pct = 0.0
    max_drawdown_pct = 0.0

    for t in trades:
        running_capital = t.capital_after
        # Track peak for max gain
        if running_capital > peak_capital:
            peak_capital = running_capital
            max_gain_pct = (peak_capital - initial_capital) / initial_capital * 100
        # Track lowest for max drawdown
        if running_capital < lowest_capital:
            lowest_capital = running_capital

    # Max drawdown is how far below initial capital we went (as positive number for display)
    if lowest_capital < initial_capital:
        max_drawdown_pct = (initial_capital - lowest_capital) / initial_capital * 100

    total_return_pct = (running_capital - initial_capital) / initial_capital * 100 if trades else 0

    # Count signals per model with LONG/SHORT breakdown
    signal_counts = {model: len(sigs) for model, sigs in signals.items()}

    # Calculate detailed signal statistics
    signal_stats = {}
    # Estimate trading days from OHLCV data
    trading_days = max(1, len(ohlcv_df) / 390)  # ~390 minutes per trading day

    for model, signal_list in signals.items():
        long_signals = sum(1 for s in signal_list if s.signal_type == "long")
        short_signals = sum(1 for s in signal_list if s.signal_type == "short")
        total = long_signals + short_signals
        signal_stats[model] = {
            "total": total,
            "long": long_signals,
            "short": short_signals,
            "per_day": total / trading_days if trading_days > 0 else 0,
            "long_per_day": long_signals / trading_days if trading_days > 0 else 0,
            "short_per_day": short_signals / trading_days if trading_days > 0 else 0,
            "long_pct": long_signals / total * 100 if total > 0 else 0,
            "short_pct": short_signals / total * 100 if total > 0 else 0,
        }

    # Build signal statistics HTML
    signal_stats_rows = ""
    signal_display_names = {"1m": "C1", "5m": "C5", "15m": "C15", "breach": "B", "bounce": "V", "news_event": "NE"}
    for model in ["1m", "5m", "15m", "breach", "bounce", "news_event"]:
        if model in signal_stats:
            s = signal_stats[model]
            display_name = signal_display_names.get(model, model.upper())
            signal_stats_rows += f"""
                <tr>
                    <td><strong>{display_name}</strong></td>
                    <td>{s['total']:,}</td>
                    <td class="long">{s['long']:,} ({s['long_pct']:.1f}%)</td>
                    <td class="short">{s['short']:,} ({s['short_pct']:.1f}%)</td>
                    <td>{s['per_day']:.1f}</td>
                    <td class="long">{s['long_per_day']:.1f}</td>
                    <td class="short">{s['short_per_day']:.1f}</td>
                </tr>
            """

    # Create trade time ranges for candle highlighting
    trade_ranges = []
    for i, t in enumerate(trades):
        trade_ranges.append({
            "idx": i,
            "entry": t.entry_time,
            "exit": t.exit_time,
        })

    # Create balance history data for different timeframe views
    # 1. Trade-level: green for winning trades, red for losing trades
    # 2. Daily: green for positive days, red for negative days
    # 3. Weekly: green for positive weeks, red for negative weeks

    # Sharpe ratio (calculated from daily returns, using SPY returns as benchmark)
    sharpe_ratio = 0.0
    daily_pnl = {}
    daily_end_balance = {}

    # Calculate SPY daily returns from OHLCV data
    spy_daily_returns = {}
    if not ohlcv_df.empty:
        # Group by date and calculate daily returns
        ohlcv_with_date = ohlcv_df.copy()
        ohlcv_with_date['date'] = ohlcv_with_date.index.date
        for date_key, group in ohlcv_with_date.groupby('date'):
            day_open = group['open'].iloc[0]
            day_close = group['close'].iloc[-1]
            if day_open > 0:
                spy_daily_returns[date_key] = ((day_close - day_open) / day_open) * 100

    if trades:
        initial_cap = trades[0].capital_after - trades[0].pnl_dollars

        # --- Daily data for Sharpe calculation ---
        for t in trades:
            trade_time = datetime.fromtimestamp(t.exit_time)
            trade_date = trade_time.date()
            if trade_date not in daily_pnl:
                daily_pnl[trade_date] = 0.0
            daily_pnl[trade_date] += t.pnl_dollars
            daily_end_balance[trade_date] = t.capital_after

        # Calculate Sharpe Ratio from daily returns (excess over SPY)
        if len(daily_pnl) >= 2:
            # Calculate daily returns as percentage of capital at start of each day
            daily_returns = []
            daily_excess_returns = []
            sorted_dates = sorted(daily_end_balance.keys())
            prev_balance = initial_cap
            for date_key in sorted_dates:
                day_return_pct = (daily_pnl[date_key] / prev_balance) * 100
                daily_returns.append(day_return_pct)
                # Calculate excess return over SPY for this day
                spy_return = spy_daily_returns.get(date_key, 0.0)
                daily_excess_returns.append(day_return_pct - spy_return)
                prev_balance = daily_end_balance[date_key]

            if len(daily_excess_returns) >= 2:
                import statistics
                mean_excess_return = statistics.mean(daily_excess_returns)
                std_excess_return = statistics.stdev(daily_excess_returns)
                if std_excess_return > 0:
                    # Annualized Sharpe = (mean excess return / std) * sqrt(252 trading days)
                    sharpe_ratio = (mean_excess_return / std_excess_return) * (252 ** 0.5)

    # Calculate monthly P&L
    monthly_stats = {}
    for t in trades:
        month_key = datetime.fromtimestamp(t.entry_time).strftime("%Y-%m")
        if month_key not in monthly_stats:
            monthly_stats[month_key] = {
                "trades": 0, "pnl": 0.0, "wins": 0, "balance": 0.0,
                "long_trades": 0, "short_trades": 0, "long_pnl": 0.0, "short_pnl": 0.0
            }
        monthly_stats[month_key]["trades"] += 1
        monthly_stats[month_key]["pnl"] += t.pnl_dollars
        monthly_stats[month_key]["balance"] = t.capital_after
        if t.is_win:
            monthly_stats[month_key]["wins"] += 1
        if t.is_long:
            monthly_stats[month_key]["long_trades"] += 1
            monthly_stats[month_key]["long_pnl"] += t.pnl_dollars
        else:
            monthly_stats[month_key]["short_trades"] += 1
            monthly_stats[month_key]["short_pnl"] += t.pnl_dollars

    monthly_rows = ""
    for month in sorted(monthly_stats.keys()):
        m = monthly_stats[month]
        win_rate_m = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
        pnl_class = "positive" if m["pnl"] >= 0 else "negative"
        long_pnl_class = "positive" if m["long_pnl"] >= 0 else "negative"
        short_pnl_class = "positive" if m["short_pnl"] >= 0 else "negative"
        # Calculate timestamps for month range (1st day 9:30 AM - last day 4:00 PM)
        month_dt = datetime.strptime(month + "-01", "%Y-%m-%d")
        month_start_ts = int((month_dt.replace(hour=9, minute=30)).timestamp())
        # Get last day of month
        if month_dt.month == 12:
            next_month = month_dt.replace(year=month_dt.year + 1, month=1, day=1)
        else:
            next_month = month_dt.replace(month=month_dt.month + 1, day=1)
        last_day = next_month - pd.Timedelta(days=1)
        month_end_ts = int((last_day.replace(hour=16, minute=0)).timestamp())
        monthly_rows += f"""
            <tr style="cursor: pointer;" onclick="focusOnPeriod({month_start_ts}, {month_end_ts})">
                <td><strong>{month}</strong></td>
                <td>{m['trades']}</td>
                <td>{win_rate_m:.1f}%</td>
                <td class="{pnl_class}">${m['pnl']:+,.0f}</td>
                <td>{m['long_trades']}</td>
                <td class="{long_pnl_class}">${m['long_pnl']:+,.0f}</td>
                <td>{m['short_trades']}</td>
                <td class="{short_pnl_class}">${m['short_pnl']:+,.0f}</td>
                <td>${m['balance']:,.0f}</td>
            </tr>
        """

    # Calculate weekly P&L
    weekly_stats = {}
    for t in trades:
        dt = datetime.fromtimestamp(t.entry_time)
        # Get Monday of the week
        week_start = dt - pd.Timedelta(days=dt.weekday())
        week_key = week_start.strftime("%Y-%m-%d")
        if week_key not in weekly_stats:
            weekly_stats[week_key] = {
                "trades": 0, "pnl": 0.0, "wins": 0, "balance": 0.0,
                "long_trades": 0, "short_trades": 0, "long_pnl": 0.0, "short_pnl": 0.0
            }
        weekly_stats[week_key]["trades"] += 1
        weekly_stats[week_key]["pnl"] += t.pnl_dollars
        weekly_stats[week_key]["balance"] = t.capital_after
        if t.is_win:
            weekly_stats[week_key]["wins"] += 1
        if t.is_long:
            weekly_stats[week_key]["long_trades"] += 1
            weekly_stats[week_key]["long_pnl"] += t.pnl_dollars
        else:
            weekly_stats[week_key]["short_trades"] += 1
            weekly_stats[week_key]["short_pnl"] += t.pnl_dollars

    weekly_rows = ""
    for week in sorted(weekly_stats.keys()):
        w = weekly_stats[week]
        win_rate_w = w["wins"] / w["trades"] * 100 if w["trades"] > 0 else 0
        pnl_class = "positive" if w["pnl"] >= 0 else "negative"
        long_pnl_class = "positive" if w["long_pnl"] >= 0 else "negative"
        short_pnl_class = "positive" if w["short_pnl"] >= 0 else "negative"
        # Calculate timestamps for week range (Monday 9:30 AM - Friday 4:00 PM)
        week_dt = datetime.strptime(week, "%Y-%m-%d")
        week_start_ts = int((week_dt.replace(hour=9, minute=30)).timestamp())
        week_end_dt = week_dt + pd.Timedelta(days=4)  # Friday
        week_end_ts = int((week_end_dt.replace(hour=16, minute=0)).timestamp())
        weekly_rows += f"""
            <tr style="cursor: pointer;" onclick="focusOnPeriod({week_start_ts}, {week_end_ts})">
                <td><strong>{week}</strong></td>
                <td>{w['trades']}</td>
                <td>{win_rate_w:.1f}%</td>
                <td class="{pnl_class}">${w['pnl']:+,.0f}</td>
                <td>{w['long_trades']}</td>
                <td class="{long_pnl_class}">${w['long_pnl']:+,.0f}</td>
                <td>{w['short_trades']}</td>
                <td class="{short_pnl_class}">${w['short_pnl']:+,.0f}</td>
                <td>${w['balance']:,.0f}</td>
            </tr>
        """

    # Calculate daily P&L
    daily_stats = {}
    for t in trades:
        dt = datetime.fromtimestamp(t.entry_time)
        day_key = dt.strftime("%Y-%m-%d")
        if day_key not in daily_stats:
            daily_stats[day_key] = {
                "trades": 0, "pnl": 0.0, "wins": 0, "balance": 0.0,
                "long_trades": 0, "short_trades": 0, "long_pnl": 0.0, "short_pnl": 0.0
            }
        daily_stats[day_key]["trades"] += 1
        daily_stats[day_key]["pnl"] += t.pnl_dollars
        daily_stats[day_key]["balance"] = t.capital_after
        if t.is_win:
            daily_stats[day_key]["wins"] += 1
        if t.is_long:
            daily_stats[day_key]["long_trades"] += 1
            daily_stats[day_key]["long_pnl"] += t.pnl_dollars
        else:
            daily_stats[day_key]["short_trades"] += 1
            daily_stats[day_key]["short_pnl"] += t.pnl_dollars

    daily_rows = ""
    for day in sorted(daily_stats.keys()):
        d = daily_stats[day]
        win_rate_d = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0
        pnl_class = "positive" if d["pnl"] >= 0 else "negative"
        long_pnl_class = "positive" if d["long_pnl"] >= 0 else "negative"
        short_pnl_class = "positive" if d["short_pnl"] >= 0 else "negative"
        # Calculate timestamps for day range (market hours: 9:30 AM - 4:00 PM ET)
        day_dt = datetime.strptime(day, "%Y-%m-%d")
        day_start_ts = int((day_dt.replace(hour=9, minute=30)).timestamp())
        day_end_ts = int((day_dt.replace(hour=16, minute=0)).timestamp())
        daily_rows += f"""
            <tr style="cursor: pointer;" onclick="focusOnPeriod({day_start_ts}, {day_end_ts})">
                <td><strong>{day}</strong></td>
                <td>{d['trades']}</td>
                <td>{win_rate_d:.1f}%</td>
                <td class="{pnl_class}">${d['pnl']:+,.0f}</td>
                <td>{d['long_trades']}</td>
                <td class="{long_pnl_class}">${d['long_pnl']:+,.0f}</td>
                <td>{d['short_trades']}</td>
                <td class="{short_pnl_class}">${d['short_pnl']:+,.0f}</td>
                <td>${d['balance']:,.0f}</td>
            </tr>
        """

    # Calculate direction stats (calls vs puts) with VWAP breakdown
    # Categories:
    # - Long/Short (totals)
    # Direction categories with ATM/OTM breakdown:
    # - Long (Calls) - All long positions
    # - Short (Puts) - All short positions
    # - Long ATM - Long positions at-the-money (strike within $1 of entry price)
    # - Long OTM - Long positions out-of-the-money (strike > entry price for calls)
    # - Short ATM - Short positions at-the-money (strike within $1 of entry price)
    # - Short OTM - Short positions out-of-the-money (strike < entry price for puts)
    direction_categories = [
        "long",
        "short",
        "long_atm",
        "long_otm",
        "short_atm",
        "short_otm",
    ]
    direction_stats = {cat: {"trades": 0, "wins": 0, "pnl": 0.0, "avg_pnl_pct": [], "best_pnl": None, "worst_pnl": None} for cat in direction_categories}

    for t in trades:
        direction = "long" if t.is_long else "short"
        # Update overall direction stats
        direction_stats[direction]["trades"] += 1
        direction_stats[direction]["pnl"] += t.pnl_dollars
        direction_stats[direction]["avg_pnl_pct"].append(t.pnl_pct)
        if direction_stats[direction]["best_pnl"] is None or t.pnl_pct > direction_stats[direction]["best_pnl"]:
            direction_stats[direction]["best_pnl"] = t.pnl_pct
        if direction_stats[direction]["worst_pnl"] is None or t.pnl_pct < direction_stats[direction]["worst_pnl"]:
            direction_stats[direction]["worst_pnl"] = t.pnl_pct
        if t.is_win:
            direction_stats[direction]["wins"] += 1

        # Determine ATM/OTM from contract string (e.g., "SPY 600C")
        # Extract strike from contract
        strike = None
        if t.contract:
            import re
            match = re.search(r'(\d+)[CP]$', t.contract)
            if match:
                strike = float(match.group(1))

        if strike is not None:
            # ATM if strike is within $1 of entry price
            is_atm = abs(strike - t.entry_price) <= 1.0
            if t.is_long:
                strike_cat = "long_atm" if is_atm else "long_otm"
            else:
                strike_cat = "short_atm" if is_atm else "short_otm"

            direction_stats[strike_cat]["trades"] += 1
            direction_stats[strike_cat]["pnl"] += t.pnl_dollars
            direction_stats[strike_cat]["avg_pnl_pct"].append(t.pnl_pct)
            if direction_stats[strike_cat]["best_pnl"] is None or t.pnl_pct > direction_stats[strike_cat]["best_pnl"]:
                direction_stats[strike_cat]["best_pnl"] = t.pnl_pct
            if direction_stats[strike_cat]["worst_pnl"] is None or t.pnl_pct < direction_stats[strike_cat]["worst_pnl"]:
                direction_stats[strike_cat]["worst_pnl"] = t.pnl_pct
            if t.is_win:
                direction_stats[strike_cat]["wins"] += 1

    # Build direction rows with labels and descriptions
    # Structure: Long -> Long ATM, Long OTM -> Short -> Short ATM, Short OTM
    direction_labels = {
        "long": ("Long (Calls)", "All long positions"),
        "short": ("Short (Puts)", "All short positions"),
        "long_atm": ("ATM", "At-the-money (strike within $1)"),
        "long_otm": ("OTM", "Out-of-the-money (strike > price)"),
        "short_atm": ("ATM", "At-the-money (strike within $1)"),
        "short_otm": ("OTM", "Out-of-the-money (strike < price)"),
    }

    # Reorder to: long, long_atm, long_otm, short, short_atm, short_otm
    direction_order = ["long", "long_atm", "long_otm", "short", "short_atm", "short_otm"]

    direction_rows = ""
    for cat in direction_order:
        d = direction_stats[cat]
        if d["trades"] == 0:
            continue
        win_rate_d = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0
        avg_pnl = sum(d["avg_pnl_pct"]) / len(d["avg_pnl_pct"]) if d["avg_pnl_pct"] else 0
        pnl_class = "positive" if d["pnl"] >= 0 else "negative"
        label, desc = direction_labels[cat]
        # ATM/OTM rows are indented under their parent
        is_sub_row = cat in ["long_atm", "long_otm", "short_atm", "short_otm"]
        indent = "&nbsp;&nbsp;&nbsp;&nbsp;" if is_sub_row else ""
        row_style = "background: #0d0d0d;" if is_sub_row else ""
        direction_rows += f"""
            <tr style="{row_style}">
                <td>{indent}<strong>{label}</strong><br><span style="font-size: 10px; color: #888;">{indent}{desc}</span></td>
                <td>{d['trades']}</td>
                <td>{d['wins']}</td>
                <td>{win_rate_d:.1f}%</td>
                <td>{avg_pnl:+.2f}%</td>
                <td class="{pnl_class}">${d['pnl']:+,.0f}</td>
            </tr>
        """

    # Calculate time of day stats (hourly intervals from 9:30-4)
    time_intervals = [
        ("9:30-10:00", 9, 30, 10, 0),
        ("10:00-11:00", 10, 0, 11, 0),
        ("11:00-12:00", 11, 0, 12, 0),
        ("12:00-1:00", 12, 0, 13, 0),
        ("1:00-2:00", 13, 0, 14, 0),
        ("2:00-3:00", 14, 0, 15, 0),
        ("3:00-4:00", 15, 0, 16, 0),
    ]
    time_stats = {interval[0]: {"trades": 0, "wins": 0, "pnl": 0.0, "avg_pnl_pct": [], "best_pnl": None, "worst_pnl": None} for interval in time_intervals}

    for t in trades:
        dt = datetime.fromtimestamp(t.entry_time)
        hour = dt.hour
        minute = dt.minute
        # Find which interval this trade falls into
        for interval_name, start_hour, start_min, end_hour, end_min in time_intervals:
            start_minutes = start_hour * 60 + start_min
            end_minutes = end_hour * 60 + end_min
            trade_minutes = hour * 60 + minute
            if start_minutes <= trade_minutes < end_minutes:
                time_stats[interval_name]["trades"] += 1
                time_stats[interval_name]["pnl"] += t.pnl_dollars
                time_stats[interval_name]["avg_pnl_pct"].append(t.pnl_pct)
                if time_stats[interval_name]["best_pnl"] is None or t.pnl_pct > time_stats[interval_name]["best_pnl"]:
                    time_stats[interval_name]["best_pnl"] = t.pnl_pct
                if time_stats[interval_name]["worst_pnl"] is None or t.pnl_pct < time_stats[interval_name]["worst_pnl"]:
                    time_stats[interval_name]["worst_pnl"] = t.pnl_pct
                if t.is_win:
                    time_stats[interval_name]["wins"] += 1
                break

    time_rows = ""
    for interval_name, _, _, _, _ in time_intervals:
        ts = time_stats[interval_name]
        if ts["trades"] == 0:
            continue
        win_rate_t = ts["wins"] / ts["trades"] * 100 if ts["trades"] > 0 else 0
        avg_pnl = sum(ts["avg_pnl_pct"]) / len(ts["avg_pnl_pct"]) if ts["avg_pnl_pct"] else 0
        pnl_class = "positive" if ts["pnl"] >= 0 else "negative"
        time_rows += f"""
            <tr>
                <td><strong>{interval_name}</strong></td>
                <td>{ts['trades']}</td>
                <td>{ts['wins']}</td>
                <td>{win_rate_t:.1f}%</td>
                <td>{avg_pnl:+.2f}%</td>
                <td class="{pnl_class}">${ts['pnl']:+,.0f}</td>
            </tr>
        """

    # Calculate day of week stats
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    dow_stats = {day: {"trades": 0, "wins": 0, "pnl": 0.0, "avg_pnl_pct": [], "best_pnl": None, "worst_pnl": None} for day in day_names}

    for t in trades:
        dt = datetime.fromtimestamp(t.entry_time)
        weekday = dt.weekday()  # 0 = Monday, 4 = Friday
        if 0 <= weekday <= 4:  # Only weekdays
            day_name = day_names[weekday]
            dow_stats[day_name]["trades"] += 1
            dow_stats[day_name]["pnl"] += t.pnl_dollars
            dow_stats[day_name]["avg_pnl_pct"].append(t.pnl_pct)
            if dow_stats[day_name]["best_pnl"] is None or t.pnl_pct > dow_stats[day_name]["best_pnl"]:
                dow_stats[day_name]["best_pnl"] = t.pnl_pct
            if dow_stats[day_name]["worst_pnl"] is None or t.pnl_pct < dow_stats[day_name]["worst_pnl"]:
                dow_stats[day_name]["worst_pnl"] = t.pnl_pct
            if t.is_win:
                dow_stats[day_name]["wins"] += 1

    dow_rows = ""
    for day_name in day_names:
        ds = dow_stats[day_name]
        if ds["trades"] == 0:
            continue
        win_rate_d = ds["wins"] / ds["trades"] * 100 if ds["trades"] > 0 else 0
        avg_pnl = sum(ds["avg_pnl_pct"]) / len(ds["avg_pnl_pct"]) if ds["avg_pnl_pct"] else 0
        pnl_class = "positive" if ds["pnl"] >= 0 else "negative"
        dow_rows += f"""
            <tr>
                <td><strong>{day_name}</strong></td>
                <td>{ds['trades']}</td>
                <td>{ds['wins']}</td>
                <td>{win_rate_d:.1f}%</td>
                <td>{avg_pnl:+.2f}%</td>
                <td class="{pnl_class}">${ds['pnl']:+,.0f}</td>
            </tr>
        """

    # Calculate strategy stats with long/short breakdown
    # Strategies: Breach, JEPA
    strategies = ["breach", "bounce", "news_event", "jepa"]
    strategy_stats = {
        strat: {
            "total": {"trades": 0, "wins": 0, "pnl": 0.0, "avg_pnl_pct": [], "best_pnl": None, "worst_pnl": None},
            "long": {"trades": 0, "wins": 0, "pnl": 0.0, "avg_pnl_pct": [], "best_pnl": None, "worst_pnl": None},
            "short": {"trades": 0, "wins": 0, "pnl": 0.0, "avg_pnl_pct": [], "best_pnl": None, "worst_pnl": None},
        }
        for strat in strategies
    }

    for t in trades:
        # Determine strategy based on dominant_model
        if t.dominant_model == "breach":
            strategy = "breach"
        elif t.dominant_model == "bounce":
            strategy = "bounce"
        elif t.dominant_model == "news_event":
            strategy = "news_event"
        elif t.dominant_model in ("1m", "5m", "15m"):
            strategy = "jepa"
        else:
            continue  # Skip unknown strategies

        direction = "long" if t.is_long else "short"

        # Update total stats
        strategy_stats[strategy]["total"]["trades"] += 1
        strategy_stats[strategy]["total"]["pnl"] += t.pnl_dollars
        strategy_stats[strategy]["total"]["avg_pnl_pct"].append(t.pnl_pct)
        if strategy_stats[strategy]["total"]["best_pnl"] is None or t.pnl_pct > strategy_stats[strategy]["total"]["best_pnl"]:
            strategy_stats[strategy]["total"]["best_pnl"] = t.pnl_pct
        if strategy_stats[strategy]["total"]["worst_pnl"] is None or t.pnl_pct < strategy_stats[strategy]["total"]["worst_pnl"]:
            strategy_stats[strategy]["total"]["worst_pnl"] = t.pnl_pct
        if t.is_win:
            strategy_stats[strategy]["total"]["wins"] += 1

        # Update direction stats
        strategy_stats[strategy][direction]["trades"] += 1
        strategy_stats[strategy][direction]["pnl"] += t.pnl_dollars
        strategy_stats[strategy][direction]["avg_pnl_pct"].append(t.pnl_pct)
        if strategy_stats[strategy][direction]["best_pnl"] is None or t.pnl_pct > strategy_stats[strategy][direction]["best_pnl"]:
            strategy_stats[strategy][direction]["best_pnl"] = t.pnl_pct
        if strategy_stats[strategy][direction]["worst_pnl"] is None or t.pnl_pct < strategy_stats[strategy][direction]["worst_pnl"]:
            strategy_stats[strategy][direction]["worst_pnl"] = t.pnl_pct
        if t.is_win:
            strategy_stats[strategy][direction]["wins"] += 1

    # Build strategy rows with long/short breakdown
    strategy_labels = {
        "breach": ("(B)reakout", ""),
        "bounce": ("(V)WAP Bounce", "Mean reversion at VWAP support/resistance"),
        "news_event": ("(N)ews (E)vent", "High-volume continuation after opening range"),
        "jepa": ("(C)ontinuation", ""),
    }

    strategy_rows = ""
    for strat in strategies:
        stats = strategy_stats[strat]
        if stats["total"]["trades"] == 0:
            continue

        label, desc = strategy_labels[strat]
        total = stats["total"]
        win_rate = total["wins"] / total["trades"] * 100 if total["trades"] > 0 else 0
        avg_pnl = sum(total["avg_pnl_pct"]) / len(total["avg_pnl_pct"]) if total["avg_pnl_pct"] else 0
        pnl_class = "positive" if total["pnl"] >= 0 else "negative"

        # Strategy header row
        strategy_rows += f"""
            <tr style="background: #1a1a1a;">
                <td><strong>{label}</strong><br><span style="font-size: 10px; color: #888;">{desc}</span></td>
                <td><strong>{total['trades']}</strong></td>
                <td><strong>{total['wins']}</strong></td>
                <td><strong>{win_rate:.1f}%</strong></td>
                <td><strong>{avg_pnl:+.2f}%</strong></td>
                <td class="{pnl_class}"><strong>${total['pnl']:+,.0f}</strong></td>
            </tr>
        """

        # Long substrategy row
        long_stats = stats["long"]
        if long_stats["trades"] > 0:
            long_wr = long_stats["wins"] / long_stats["trades"] * 100
            long_avg = sum(long_stats["avg_pnl_pct"]) / len(long_stats["avg_pnl_pct"]) if long_stats["avg_pnl_pct"] else 0
            long_pnl_class = "positive" if long_stats["pnl"] >= 0 else "negative"
            strategy_rows += f"""
            <tr style="background: #0d1f0d;">
                <td style="padding-left: 24px; color: #26a69a;">&#x25B2; Long</td>
                <td>{long_stats['trades']}</td>
                <td>{long_stats['wins']}</td>
                <td>{long_wr:.1f}%</td>
                <td>{long_avg:+.2f}%</td>
                <td class="{long_pnl_class}">${long_stats['pnl']:+,.0f}</td>
            </tr>
            """

        # Short substrategy row
        short_stats = stats["short"]
        if short_stats["trades"] > 0:
            short_wr = short_stats["wins"] / short_stats["trades"] * 100
            short_avg = sum(short_stats["avg_pnl_pct"]) / len(short_stats["avg_pnl_pct"]) if short_stats["avg_pnl_pct"] else 0
            short_pnl_class = "positive" if short_stats["pnl"] >= 0 else "negative"
            strategy_rows += f"""
            <tr style="background: #1f0d0d;">
                <td style="padding-left: 24px; color: #ef5350;">&#x25BC; Short</td>
                <td>{short_stats['trades']}</td>
                <td>{short_stats['wins']}</td>
                <td>{short_wr:.1f}%</td>
                <td>{short_avg:+.2f}%</td>
                <td class="{short_pnl_class}">${short_stats['pnl']:+,.0f}</td>
            </tr>
            """

    # Calculate quality stats with long/short breakdown
    quality_grades = ["A", "B", "C", "D"]
    quality_labels = {
        "A": ("A — 5%", "#26a69a"),
        "B": ("B — 4%", "#66bb6a"),
        "C": ("C — 3%", "#ffa726"),
        "D": ("D — 2%", "#ef5350"),
    }
    quality_stats = {
        grade: {
            "total": {"trades": 0, "wins": 0, "pnl": 0.0, "avg_pnl_pct": []},
            "long": {"trades": 0, "wins": 0, "pnl": 0.0, "avg_pnl_pct": []},
            "short": {"trades": 0, "wins": 0, "pnl": 0.0, "avg_pnl_pct": []},
        }
        for grade in quality_grades
    }

    for t in trades:
        grade = t.quality if t.quality in quality_grades else "D"
        direction = "long" if t.is_long else "short"
        for bucket in ("total", direction):
            quality_stats[grade][bucket]["trades"] += 1
            quality_stats[grade][bucket]["pnl"] += t.pnl_dollars
            quality_stats[grade][bucket]["avg_pnl_pct"].append(t.pnl_pct)
            if t.is_win:
                quality_stats[grade][bucket]["wins"] += 1

    quality_rows = ""
    for grade in quality_grades:
        stats = quality_stats[grade]
        if stats["total"]["trades"] == 0:
            continue

        label, color = quality_labels[grade]
        total = stats["total"]
        win_rate = total["wins"] / total["trades"] * 100 if total["trades"] > 0 else 0
        avg_pnl = sum(total["avg_pnl_pct"]) / len(total["avg_pnl_pct"]) if total["avg_pnl_pct"] else 0
        pnl_class = "positive" if total["pnl"] >= 0 else "negative"

        quality_rows += f"""
            <tr style="background: #1a1a1a;">
                <td><strong style="color: {color};">{label}</strong></td>
                <td><strong>{total['trades']}</strong></td>
                <td><strong>{total['wins']}</strong></td>
                <td><strong>{win_rate:.1f}%</strong></td>
                <td><strong>{avg_pnl:+.2f}%</strong></td>
                <td class="{pnl_class}"><strong>${total['pnl']:+,.0f}</strong></td>
            </tr>
        """

        long_stats = stats["long"]
        if long_stats["trades"] > 0:
            long_wr = long_stats["wins"] / long_stats["trades"] * 100
            long_avg = sum(long_stats["avg_pnl_pct"]) / len(long_stats["avg_pnl_pct"]) if long_stats["avg_pnl_pct"] else 0
            long_pnl_class = "positive" if long_stats["pnl"] >= 0 else "negative"
            quality_rows += f"""
            <tr style="background: #0d1f0d;">
                <td style="padding-left: 24px; color: #26a69a;">&#x25B2; Long</td>
                <td>{long_stats['trades']}</td>
                <td>{long_stats['wins']}</td>
                <td>{long_wr:.1f}%</td>
                <td>{long_avg:+.2f}%</td>
                <td class="{long_pnl_class}">${long_stats['pnl']:+,.0f}</td>
            </tr>
            """

        short_stats = stats["short"]
        if short_stats["trades"] > 0:
            short_wr = short_stats["wins"] / short_stats["trades"] * 100
            short_avg = sum(short_stats["avg_pnl_pct"]) / len(short_stats["avg_pnl_pct"]) if short_stats["avg_pnl_pct"] else 0
            short_pnl_class = "positive" if short_stats["pnl"] >= 0 else "negative"
            quality_rows += f"""
            <tr style="background: #1f0d0d;">
                <td style="padding-left: 24px; color: #ef5350;">&#x25BC; Short</td>
                <td>{short_stats['trades']}</td>
                <td>{short_stats['wins']}</td>
                <td>{short_wr:.1f}%</td>
                <td>{short_avg:+.2f}%</td>
                <td class="{short_pnl_class}">${short_stats['pnl']:+,.0f}</td>
            </tr>
            """

    # Generate missed signals table (triple confluence that didn't result in trades)
    missed_signals_html = ""
    if missed_signals:
        # Count by rejection reason
        reason_counts = {}
        for ms in missed_signals:
            reason = ms.get("rejection_reason", "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        reason_summary = "<br>".join([f"{reason}: {count}" for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])])

        missed_rows = ""
        for ms in missed_signals[:50]:  # Show first 50
            et_time = ms.get("et_time", "N/A")
            signal_type = ms.get("signal_type", "?").upper()
            price = ms.get("price", 0)
            reason = ms.get("rejection_reason", "unknown")
            models = ", ".join(ms.get("agreeing_models", []))
            signal_class = "positive" if signal_type == "LONG" else "negative"
            missed_rows += f"""
                <tr>
                    <td>{et_time}</td>
                    <td class="{signal_class}">{signal_type}</td>
                    <td>${price:.2f}</td>
                    <td>{models}</td>
                    <td>{reason}</td>
                </tr>
            """

        missed_signals_html = f"""
            <div class="details-block">
                <h3>Missed Triple-Confluence Signals ({len(missed_signals)} total)</h3>
                <p style="color: #888; font-size: 11px; margin-bottom: 10px;">
                    Signals where all 3 models agreed but no trade was taken.<br>
                    <strong>Reasons:</strong> {reason_summary}
                </p>
                <table class="trades-table" style="font-size: 11px;">
                    <thead>
                        <tr>
                            <th>Time (ET)</th>
                            <th>Signal</th>
                            <th>Price</th>
                            <th>Models</th>
                            <th>Rejection Reason</th>
                        </tr>
                    </thead>
                    <tbody>
                        {missed_rows}
                    </tbody>
                </table>
                {'<p style="color: #888; font-size: 10px;">Showing first 50 of ' + str(len(missed_signals)) + '</p>' if len(missed_signals) > 50 else ''}
            </div>
        """

    # Extract date range info from OHLCV data
    if len(ohlcv_df) > 0:
        first_ts = ohlcv_df.index[0]
        last_ts = ohlcv_df.index[-1]
        start_date_str = first_ts.strftime("%Y-%m-%d")
        end_date_str = last_ts.strftime("%Y-%m-%d")
        # Calculate span in days for conditional tab display
        date_span_days = (last_ts - first_ts).days
    else:
        start_date_str = "N/A"
        end_date_str = "N/A"
        date_span_days = 0

    # Determine which tabs to show based on date range
    show_daily_tab = date_span_days >= 1
    show_weekly_tab = date_span_days >= 7
    show_monthly_tab = date_span_days >= 30
    show_yearly_tab = date_span_days >= 365

    # Calculate yearly P&L (only if we have >1 year of data)
    yearly_stats = {}
    if show_yearly_tab:
        for t in trades:
            dt = datetime.fromtimestamp(t.entry_time)
            year_key = str(dt.year)
            if year_key not in yearly_stats:
                yearly_stats[year_key] = {
                    "trades": 0, "pnl": 0.0, "wins": 0, "balance": 0.0,
                    "long_trades": 0, "short_trades": 0, "long_pnl": 0.0, "short_pnl": 0.0
                }
            yearly_stats[year_key]["trades"] += 1
            yearly_stats[year_key]["pnl"] += t.pnl_dollars
            yearly_stats[year_key]["balance"] = t.capital_after
            if t.is_win:
                yearly_stats[year_key]["wins"] += 1
            if t.is_long:
                yearly_stats[year_key]["long_trades"] += 1
                yearly_stats[year_key]["long_pnl"] += t.pnl_dollars
            else:
                yearly_stats[year_key]["short_trades"] += 1
                yearly_stats[year_key]["short_pnl"] += t.pnl_dollars

    yearly_rows = ""
    for year in sorted(yearly_stats.keys()):
        y = yearly_stats[year]
        win_rate_y = y["wins"] / y["trades"] * 100 if y["trades"] > 0 else 0
        pnl_class = "positive" if y["pnl"] >= 0 else "negative"
        long_pnl_class = "positive" if y["long_pnl"] >= 0 else "negative"
        short_pnl_class = "positive" if y["short_pnl"] >= 0 else "negative"
        # Calculate timestamps for year range (Jan 1 9:30 AM - Dec 31 4:00 PM)
        year_start_dt = datetime(int(year), 1, 1, 9, 30)
        year_end_dt = datetime(int(year), 12, 31, 16, 0)
        year_start_ts = int(year_start_dt.timestamp())
        year_end_ts = int(year_end_dt.timestamp())
        yearly_rows += f"""
            <tr style="cursor: pointer;" onclick="focusOnPeriod({year_start_ts}, {year_end_ts})">
                <td><strong>{year}</strong></td>
                <td>{y['trades']}</td>
                <td>{win_rate_y:.1f}%</td>
                <td class="{pnl_class}">${y['pnl']:+,.0f}</td>
                <td>{y['long_trades']}</td>
                <td class="{long_pnl_class}">${y['long_pnl']:+,.0f}</td>
                <td>{y['short_trades']}</td>
                <td class="{short_pnl_class}">${y['short_pnl']:+,.0f}</td>
                <td>${y['balance']:,.0f}</td>
            </tr>
        """

    # Get test timestamp
    test_timestamp = datetime.now().strftime("%Y-%m-%d at %I:%M %p")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - {combination_name}</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        html, body {{
            height: 100%;
            overflow: hidden;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0a0a0a;
            color: #eee;
            display: flex;
            flex-direction: column;
        }}
        /* Section 1: Header - thin row */
        .header {{
            height: 40px;
            min-height: 40px;
            background: #111;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            padding: 0 20px;
            border-bottom: 1px solid #333;
        }}
        .header-item {{
            color: #aaa;
            font-size: 13px;
        }}
        .header-item strong {{
            color: #26a69a;
        }}
        /* Section 2: Summary cards */
        .summary {{
            height: 80px;
            min-height: 80px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 30px;
            padding: 10px 20px;
            background: #111;
            flex-wrap: wrap;
        }}
        .stat {{
            text-align: center;
            padding: 8px 20px;
            background: #1a1a1a;
            border-radius: 6px;
        }}
        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #26a69a;
        }}
        .stat-value.positive {{
            color: #26a69a;
        }}
        .stat-value.negative {{
            color: #ef5350;
        }}
        .stat-value .positive {{
            color: #26a69a;
        }}
        .stat-value .negative {{
            color: #ef5350;
        }}
        .stat-label {{
            font-size: 11px;
            color: #888;
            margin-top: 2px;
        }}
        /* Section 3: Chart (~45% of remaining) */
        .chart-section {{
            flex: 63;
            min-height: 300px;
            display: flex;
            flex-direction: column;
            background: #0a0a0a;
            border-bottom: 1px solid #333;
        }}
        .legend {{
            height: 40px;
            min-height: 40px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            flex-wrap: wrap;
            padding: 5px 10px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            background: #1a1a1a;
            border-radius: 4px;
            cursor: pointer;
            user-select: none;
            transition: opacity 0.2s;
            font-size: 12px;
        }}
        .legend-item:hover {{
            background: #2a2a2a;
        }}
        .legend-item.disabled {{
            opacity: 0.4;
        }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}
        .chart-wrapper {{
            flex: 1;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }}
        #chart {{
            flex: 1;
            min-height: 0;
        }}
        #volume-chart {{
            height: 60px;
            min-height: 60px;
        }}
        /* Section 4: Table (~55% of remaining) */
        .table-section {{
            flex: 37;
            min-height: 200px;
            display: flex;
            flex-direction: column;
            background: #111;
            overflow: hidden;
        }}
        .tabs {{
            height: 40px;
            min-height: 40px;
            display: flex;
            border-bottom: 1px solid #333;
        }}
        .tab {{
            padding: 0 25px;
            cursor: pointer;
            background: transparent;
            border: none;
            color: #888;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
            display: flex;
            align-items: center;
        }}
        .tab:hover {{
            color: #fff;
            background: rgba(255,255,255,0.05);
        }}
        .tab.active {{
            color: #26a69a;
            border-bottom: 2px solid #26a69a;
        }}
        .tab-content {{
            display: none;
            flex: 1;
            overflow: auto;
            padding: 10px 15px;
        }}
        .tab-content.active {{
            display: flex;
            flex-direction: column;
        }}
        .trades-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        .trades-table th {{
            background: #1a1a1a;
            padding: 8px 6px;
            text-align: left;
            position: sticky;
            top: -10px;
            z-index: 10;
        }}
        .trades-table td {{
            padding: 6px;
            border-bottom: 1px solid #333;
        }}
        .trades-table tr:hover {{
            background: #2a2a2a;
        }}
        .trades-table tr.win {{
            background: rgba(38, 166, 154, 0.1);
        }}
        .trades-table tr.loss {{
            background: rgba(239, 83, 80, 0.1);
        }}
        .trades-table tr.runner td:first-child::after {{
            content: " (R)";
            color: #ff9800;
            font-size: 9px;
        }}
        .trades-table .pnl {{
            font-weight: bold;
        }}
        .trades-table tr.win .pnl {{
            color: #26a69a;
        }}
        .trades-table tr.loss .pnl {{
            color: #ef5350;
        }}
        .trades-table td.positive {{
            color: #26a69a;
            font-weight: bold;
        }}
        .trades-table td.negative {{
            color: #ef5350;
            font-weight: bold;
        }}
        .trades-table tr.selected {{
            background: rgba(38, 166, 154, 0.3) !important;
            box-shadow: inset 0 0 0 2px #26a69a;
        }}
        .details-section {{
            display: grid;
            gap: 15px;
        }}
        .details-block {{
            background: #1a1a1a;
            padding: 12px;
            border-radius: 6px;
        }}
        .details-block h3 {{
            color: #26a69a;
            margin-bottom: 8px;
            font-size: 13px;
        }}
        .details-block pre {{
            background: #0a0a0a;
            padding: 8px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 11px;
            color: #ddd;
            white-space: pre-wrap;
            word-break: break-all;
        }}
        .details-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
        }}
        .detail-item {{
            background: #0a0a0a;
            padding: 10px;
            border-radius: 4px;
        }}
        .detail-label {{
            color: #888;
            font-size: 10px;
            margin-bottom: 3px;
        }}
        .detail-value {{
            color: #fff;
            font-size: 13px;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <!-- Section 1: Header -->
    <div class="header">
        <span class="header-item"><strong>SPY</strong>: {start_date_str} → {end_date_str}</span>
        <span class="header-item"><strong>{int(trading_days)}</strong> trading days</span>
        <span class="header-item">Tested: <strong>{test_timestamp}</strong></span>
    </div>

    <!-- Section 2: Summary Cards -->
    <div class="summary">
        <div class="stat">
            <div class="stat-value">{total_trades}</div>
            <div class="stat-label">Total Trades</div>
        </div>
        <div class="stat">
            <div class="stat-value">{win_rate:.1f}%</div>
            <div class="stat-label">Win Rate</div>
        </div>
        <div class="stat">
            <div class="stat-value {'positive' if total_return_pct >= 0 else 'negative'}">{total_return_pct:+.1f}%</div>
            <div class="stat-label">Total Return</div>
        </div>
        <div class="stat">
            <div class="stat-value"><span class="positive">+{max_gain_pct:.1f}%</span><span style="color: #888;"> / </span><span class="negative">-{max_drawdown_pct:.1f}%</span></div>
            <div class="stat-label">Best/Worst Alltime P/L</div>
        </div>
        <div class="stat">
            <div class="stat-value"><span class="positive">{metrics.get('best_daily_pnl_pct', 0):+.2f}%</span><span style="color: #888;"> / </span><span class="negative">{metrics.get('worst_daily_pnl_pct', 0):+.2f}%</span></div>
            <div class="stat-label">Best/Worst Daily P/L</div>
        </div>
        <div class="stat">
            <div class="stat-value"><span class="positive">{metrics.get('best_weekly_pnl_pct', 0):+.2f}%</span><span style="color: #888;"> / </span><span class="negative">{metrics.get('worst_weekly_pnl_pct', 0):+.2f}%</span></div>
            <div class="stat-label">Best/Worst Weekly P/L</div>
        </div>
        <div class="stat">
            <div class="stat-value {'positive' if metrics.get('daily_win_rate', 0) >= 50 else 'negative'}">{metrics.get('daily_win_rate', 0):.1f}%</div>
            <div class="stat-label">Daily Win ({metrics.get('green_days', 0)}/{metrics.get('total_days', 0)})</div>
        </div>
        <div class="stat">
            <div class="stat-value {'positive' if metrics.get('weekly_win_rate', 0) >= 50 else 'negative'}">{metrics.get('weekly_win_rate', 0):.1f}%</div>
            <div class="stat-label">Weekly Win ({metrics.get('green_weeks', 0)}/{metrics.get('total_weeks', 0)})</div>
        </div>
        <div class="stat">
            <div class="stat-value {'positive' if sharpe_ratio >= 1.0 else 'negative' if sharpe_ratio < 0 else ''}">{sharpe_ratio:.2f}</div>
            <div class="stat-label">Sharpe Ratio</div>
        </div>
    </div>

    <!-- Section 3: Chart -->
    <div class="chart-section">
        <div class="legend">
            <div class="legend-item timeframe-indicator" id="timeframe-indicator" style="background: #2a2a2a; cursor: default;">
                <span>Candles: <strong id="current-tf">1m</strong></span>
            </div>
            <div class="legend-item" data-series="trades" onclick="toggleSeries('trades')">
                <div class="legend-color" style="background: linear-gradient(135deg, #26a69a 50%, #ef5350 50%);"></div>
                <span>Trades ({total_trades})</span>
            </div>
            <div class="legend-item" data-series="1m" onclick="toggleSeries('1m')">
                <div class="legend-color" style="background: #ffeb3b;"></div>
                <span>C1 ({signal_counts.get('1m', 0)})</span>
            </div>
            <div class="legend-item" data-series="5m" onclick="toggleSeries('5m')">
                <div class="legend-color" style="background: #ff9800;"></div>
                <span>C5 ({signal_counts.get('5m', 0)})</span>
            </div>
            <div class="legend-item" data-series="15m" onclick="toggleSeries('15m')">
                <div class="legend-color" style="background: #9c27b0;"></div>
                <span>C15 ({signal_counts.get('15m', 0)})</span>
            </div>
            <div class="legend-item" data-series="breach" onclick="toggleSeries('breach')">
                <div class="legend-color" style="background: #00bcd4;"></div>
                <span>B ({signal_counts.get('breach', 0)})</span>
            </div>
            <div class="legend-item" data-series="bounce" onclick="toggleSeries('bounce')">
                <div class="legend-color" style="background: #4caf50;"></div>
                <span>V ({signal_counts.get('bounce', 0)})</span>
            </div>
            <div class="legend-item" data-series="news_event" onclick="toggleSeries('news_event')">
                <div class="legend-color" style="background: #ffc107;"></div>
                <span>NE ({signal_counts.get('news_event', 0)})</span>
            </div>
            <div class="legend-item" data-series="vwap" onclick="toggleSeries('vwap')">
                <div class="legend-color" style="background: linear-gradient(to bottom, #FFFFFF 40%, #8c8c8c 60%); border: 1px solid #666;"></div>
                <span>VWAP</span>
            </div>
        </div>
        <div class="chart-wrapper">
            <div id="chart"></div>
            <div id="volume-chart"></div>
        </div>
    </div>

    <!-- Section 4: Table -->
    <div class="table-section">
        <div class="tabs">
            <button class="tab active" onclick="switchTab('trades')">Trades</button>
            {'<button class="tab" onclick="switchTab(\'daily\')">By Day</button>' if show_daily_tab else ''}
            {'<button class="tab" onclick="switchTab(\'weekly\')">By Week</button>' if show_weekly_tab else ''}
            {'<button class="tab" onclick="switchTab(\'monthly\')">By Month</button>' if show_monthly_tab else ''}
            {'<button class="tab" onclick="switchTab(\'yearly\')">By Year</button>' if show_yearly_tab else ''}
            <button class="tab" onclick="switchTab('direction')">By Direction</button>
            <button class="tab" onclick="switchTab('strategy')">By Strategy</button>
            <button class="tab" onclick="switchTab('quality')">By Quality</button>
            <button class="tab" onclick="switchTab('timeofday')">By Time of Day</button>
            <button class="tab" onclick="switchTab('dayofweek')">By Day of Week</button>
            <button class="tab" onclick="switchTab('details')">Details</button>
        </div>
        <div id="trades-tab" class="tab-content active">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Entry</th>
                        <th>Duration</th>
                        <th>Contract</th>
                        <th>Qty</th>
                        <th>Entry $</th>
                        <th>Exit $</th>
                        <th>P&L %</th>
                        <th>P&L $</th>
                        <th>Balance</th>
                        <th>Quality</th>
                        <th>Entry Reason</th>
                        <th>Exit Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {trades_table_html}
                </tbody>
            </table>
        </div>
        {"" if not show_daily_tab else '''<div id="daily-tab" class="tab-content">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>P&L</th>
                        <th>Long</th>
                        <th>Long P/L</th>
                        <th>Short</th>
                        <th>Short P/L</th>
                        <th>Balance</th>
                    </tr>
                </thead>
                <tbody>
                    ''' + daily_rows + '''
                </tbody>
            </table>
        </div>'''}
        {"" if not show_weekly_tab else '''<div id="weekly-tab" class="tab-content">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Week Start</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>P&L</th>
                        <th>Long</th>
                        <th>Long P/L</th>
                        <th>Short</th>
                        <th>Short P/L</th>
                        <th>Balance</th>
                    </tr>
                </thead>
                <tbody>
                    ''' + weekly_rows + '''
                </tbody>
            </table>
        </div>'''}
        {"" if not show_monthly_tab else '''<div id="monthly-tab" class="tab-content">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Month</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>P&L</th>
                        <th>Long</th>
                        <th>Long P/L</th>
                        <th>Short</th>
                        <th>Short P/L</th>
                        <th>Balance</th>
                    </tr>
                </thead>
                <tbody>
                    ''' + monthly_rows + '''
                </tbody>
            </table>
        </div>'''}
        {"" if not show_yearly_tab else '''<div id="yearly-tab" class="tab-content">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Year</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>P&L</th>
                        <th>Long</th>
                        <th>Long P/L</th>
                        <th>Short</th>
                        <th>Short P/L</th>
                        <th>Balance</th>
                    </tr>
                </thead>
                <tbody>
                    ''' + yearly_rows + '''
                </tbody>
            </table>
        </div>'''}
        <div id="direction-tab" class="tab-content">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Direction</th>
                        <th>Trades</th>
                        <th>Wins</th>
                        <th>Win Rate</th>
                        <th>Avg P/L</th>
                        <th>Total P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {direction_rows}
                </tbody>
            </table>
        </div>
        <div id="strategy-tab" class="tab-content">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Strategy</th>
                        <th>Trades</th>
                        <th>Wins</th>
                        <th>Win Rate</th>
                        <th>Avg P/L</th>
                        <th>Total P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {strategy_rows}
                </tbody>
            </table>
        </div>
        <div id="quality-tab" class="tab-content">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Quality</th>
                        <th>Trades</th>
                        <th>Wins</th>
                        <th>Win Rate</th>
                        <th>Avg P/L</th>
                        <th>Total P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {quality_rows}
                </tbody>
            </table>
        </div>
        <div id="timeofday-tab" class="tab-content">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Time Interval</th>
                        <th>Trades</th>
                        <th>Wins</th>
                        <th>Win Rate</th>
                        <th>Avg P/L</th>
                        <th>Total P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {time_rows}
                </tbody>
            </table>
        </div>
        <div id="dayofweek-tab" class="tab-content">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Day</th>
                        <th>Trades</th>
                        <th>Wins</th>
                        <th>Win Rate</th>
                        <th>Avg P/L</th>
                        <th>Total P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {dow_rows}
                </tbody>
            </table>
        </div>
        <div id="details-tab" class="tab-content">
            <div class="details-section">
                {f'<div class="details-block"><h3>Command Used</h3><pre>{command}</pre></div>' if command else ''}
                <div class="details-block">
                    <h3>Signal Statistics</h3>
                    <table class="trades-table" style="font-size: 11px;">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Total</th>
                                <th>Long</th>
                                <th>Short</th>
                                <th>/Day</th>
                                <th>Long/Day</th>
                                <th>Short/Day</th>
                            </tr>
                        </thead>
                        <tbody>
                            {signal_stats_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Data - defined first so all functions can access them
        // Multi-timeframe OHLCV data
        const ohlcvData1m = {json.dumps(ohlcv_1m)};
        const volumeData1m = {json.dumps(volume_1m)};
        const ohlcvData5m = {json.dumps(ohlcv_5m)};
        const volumeData5m = {json.dumps(volume_5m)};
        const ohlcvData15m = {json.dumps(ohlcv_15m)};
        const volumeData15m = {json.dumps(volume_15m)};
        const ohlcvData1h = {json.dumps(ohlcv_1h)};
        const volumeData1h = {json.dumps(volume_1h)};
        const ohlcvData1d = {json.dumps(ohlcv_1d)};
        const volumeData1d = {json.dumps(volume_1d)};
        const ohlcvData1w = {json.dumps(ohlcv_1w)};
        const volumeData1w = {json.dumps(volume_1w)};

        // VWAP data with bands (daily-anchored, calculated from 1m data)
        const vwapData = {json.dumps(vwap_data['vwap'])};
        const vwapUpper1sd = {json.dumps(vwap_data['upper_1sd'])};
        const vwapLower1sd = {json.dumps(vwap_data['lower_1sd'])};
        const vwapUpper2sd = {json.dumps(vwap_data['upper_2sd'])};
        const vwapLower2sd = {json.dumps(vwap_data['lower_2sd'])};

        // Default to 1m for backwards compatibility
        let ohlcvData = ohlcvData1m;
        let volumeData = volumeData1m;
        let currentTimeframe = '1m';

        const dailyBoundaries = {json.dumps(daily_boundaries)};
        const tradeEntries = {json.dumps(trade_entries)};
        const tradeExits = {json.dumps(trade_exits)};
        const tradeRanges = {json.dumps(trade_ranges)};
        const signals1m = {json.dumps(signals_json.get('1m', []))};
        const signals5m = {json.dumps(signals_json.get('5m', []))};
        const signals15m = {json.dumps(signals_json.get('15m', []))};
        const signalsBreach = {json.dumps(signals_json.get('breach', []))};
        const signalsBounce = {json.dumps(signals_json.get('bounce', []))};
        const signalsNewsEvent = {json.dumps(signals_json.get('news_event', []))};

        // Track series visibility - MUST be defined before chart creation
        // Signals hidden by default, trades and VWAP visible
        const seriesState = {{
            trades: true,
            '1m': false,
            '5m': false,
            '15m': false,
            breach: false,
            bounce: false,
            news_event: false,
            vwap: true,  // Controls both VWAP line and bands
        }};

        // Track selected trade - MUST be defined before chart creation
        let selectedTradeIdx = null;

        // Chart and series references (will be set after chart creation)
        let chart = null;
        let volumeChart = null;
        let candleSeries = null;
        let volumeSeries = null;
        let vwapSeries = null;
        let vwapUpper1sdSeries = null;
        let vwapLower1sdSeries = null;
        let vwapUpper2sdSeries = null;
        let vwapLower2sdSeries = null;
        let dayLineSeries = null;
        let dayBoundaryPrimitive = null;  // Day boundary vertical lines primitive

        // Timeframe switching based on visible range
        // Thresholds: 1m until 48h visible, then 5m until 8d, etc.
        function getTimeframeForRange(visibleSeconds) {{
            const oneDay = 24 * 60 * 60;
            const seventyTwoHours = 3 * oneDay;  // 72 hours = 259200 seconds
            const eightDays = 8 * oneDay;
            const thirtyDays = 30 * oneDay;
            const ninetyDays = 90 * oneDay;
            const oneYear = 365 * oneDay;

            if (visibleSeconds < seventyTwoHours) {{
                return '1m';  // Less than 72h visible -> 1m candles
            }} else if (visibleSeconds < eightDays) {{
                return '5m';
            }} else if (visibleSeconds < thirtyDays) {{
                return '15m';
            }} else if (visibleSeconds < ninetyDays) {{
                return '1h';
            }} else if (visibleSeconds < oneYear) {{
                return '1d';
            }} else {{
                return '1w';
            }}
        }}

        function getDataForTimeframe(tf) {{
            switch(tf) {{
                case '1m': return {{ ohlcv: ohlcvData1m, volume: volumeData1m }};
                case '5m': return {{ ohlcv: ohlcvData5m, volume: volumeData5m }};
                case '15m': return {{ ohlcv: ohlcvData15m, volume: volumeData15m }};
                case '1h': return {{ ohlcv: ohlcvData1h, volume: volumeData1h }};
                case '1d': return {{ ohlcv: ohlcvData1d, volume: volumeData1d }};
                case '1w': return {{ ohlcv: ohlcvData1w, volume: volumeData1w }};
                default: return {{ ohlcv: ohlcvData1m, volume: volumeData1m }};
            }}
        }}

        function switchTimeframe(newTf) {{
            if (newTf === currentTimeframe) return;

            const data = getDataForTimeframe(newTf);
            currentTimeframe = newTf;
            ohlcvData = data.ohlcv;
            volumeData = data.volume;

            // Update indicator
            const tfIndicator = document.getElementById('current-tf');
            if (tfIndicator) {{
                tfIndicator.textContent = newTf;
            }}

            // Toggle day boundary visibility (hide for daily or higher timeframes)
            if (dayBoundaryPrimitive) {{
                const showDayBoundaries = ['1m', '5m', '15m', '1h'].includes(newTf);
                dayBoundaryPrimitive.setVisible(showDayBoundaries);
            }}

            if (candleSeries && volumeSeries) {{
                candleSeries.setData(ohlcvData);
                volumeSeries.setData(volumeData);
                updateMarkers();
            }}
        }}

        // Focus chart on a specific time period (for day/week/month selection)
        function focusOnPeriod(startTs, endTs) {{
            if (!chart) return;

            // Clear trade selection
            selectedTradeIdx = null;
            document.querySelectorAll('.trades-table tr.selected').forEach(tr => {{
                tr.classList.remove('selected');
            }});

            try {{
                chart.timeScale().setVisibleRange({{
                    from: startTs,
                    to: endTs,
                }});
                // Let timeframe auto-switch based on range
                setTimeout(() => {{
                    chart.priceScale('right').applyOptions({{ autoScale: false }});
                    chart.priceScale('right').applyOptions({{ autoScale: true }});
                }}, 50);
            }} catch (e) {{
                console.error('Failed to focus on period:', e);
            }}

            updateMarkers();
        }}


        // Tab switching
        function switchTab(tabName) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`[onclick="switchTab('${{tabName}}')"]`).classList.add('active');
            document.getElementById(`${{tabName}}-tab`).classList.add('active');
        }}

        // Function to update markers based on visibility
        // Defined early so toggleSeries can call it
        function updateMarkers() {{
            if (!candleSeries) return; // Guard against early calls before chart init

            // Get candle duration for current timeframe (in seconds)
            function getCandleDuration(tf) {{
                switch(tf) {{
                    case '1m': return 60;
                    case '5m': return 5 * 60;
                    case '15m': return 15 * 60;
                    case '1h': return 60 * 60;
                    case '1d': return 24 * 60 * 60;
                    case '1w': return 7 * 24 * 60 * 60;
                    default: return 60;
                }}
            }}

            // Aggregate signals to show max 1 per model per direction per candle
            function aggregateSignals(signals, color, candleDuration) {{
                if (candleDuration <= 60) {{
                    // 1m timeframe - show all signals
                    return signals.map(m => ({{
                        ...m,
                        color: color,
                        shape: m.position === 'belowBar' ? 'arrowUp' : 'arrowDown',
                        size: 1,
                    }}));
                }}

                // For higher timeframes, group by candle bucket and direction
                // Key = candleBucket + direction
                const seen = new Set();
                const aggregated = [];

                for (const m of signals) {{
                    const bucket = Math.floor(m.time / candleDuration) * candleDuration;
                    const direction = m.position === 'belowBar' ? 'long' : 'short';
                    const key = `${{bucket}}_${{direction}}`;

                    if (!seen.has(key)) {{
                        seen.add(key);
                        aggregated.push({{
                            ...m,
                            time: bucket, // Align to candle start
                            color: color,
                            shape: m.position === 'belowBar' ? 'arrowUp' : 'arrowDown',
                            size: 1,
                        }});
                    }}
                }}

                return aggregated;
            }}

            const candleDuration = getCandleDuration(currentTimeframe);
            let allMarkers = [];

            if (seriesState['1m']) {{
                allMarkers.push(...aggregateSignals(signals1m, '#ffeb3b', candleDuration));
            }}

            if (seriesState['5m']) {{
                allMarkers.push(...aggregateSignals(signals5m, '#ff9800', candleDuration));
            }}

            if (seriesState['15m']) {{
                allMarkers.push(...aggregateSignals(signals15m, '#9c27b0', candleDuration));
            }}

            if (seriesState['breach']) {{
                allMarkers.push(...aggregateSignals(signalsBreach, '#00bcd4', candleDuration));
            }}

            if (seriesState['bounce']) {{
                allMarkers.push(...aggregateSignals(signalsBounce, '#4caf50', candleDuration));
            }}

            if (seriesState['news_event']) {{
                allMarkers.push(...aggregateSignals(signalsNewsEvent, '#ffc107', candleDuration));
            }}

            // Sort by time and set markers
            allMarkers.sort((a, b) => a.time - b.time);
            candleSeries.setMarkers(allMarkers);

            // Update custom primitive for trades only
            if (window.tradeMarkerPrimitive) {{
                window.tradeMarkerPrimitive.updateState(seriesState.trades, selectedTradeIdx);
            }}

            // Update candle colors for selection highlighting (white border)
            if (selectedTradeIdx !== null && tradeRanges[selectedTradeIdx]) {{
                const trade = tradeRanges[selectedTradeIdx];
                const highlightedData = ohlcvData.map(candle => {{
                    if (candle.time >= trade.entry && candle.time <= trade.exit) {{
                        return {{
                            ...candle,
                            borderColor: '#ffffff',
                            wickColor: '#ffffff',
                        }};
                    }}
                    return candle;
                }});
                candleSeries.setData(highlightedData);
            }} else {{
                candleSeries.setData(ohlcvData);
            }}
        }}

        // Toggle series visibility
        function toggleSeries(seriesName) {{
            seriesState[seriesName] = !seriesState[seriesName];
            const legendItem = document.querySelector(`[data-series="${{seriesName}}"]`);
            if (legendItem) {{
                legendItem.classList.toggle('disabled', !seriesState[seriesName]);
            }}

            // Handle VWAP series visibility (controls both line and bands)
            if (seriesName === 'vwap') {{
                const vwapVisible = seriesState.vwap;
                if (vwapSeries) vwapSeries.applyOptions({{ visible: vwapVisible }});
                if (vwapUpper1sdSeries) vwapUpper1sdSeries.applyOptions({{ visible: vwapVisible }});
                if (vwapLower1sdSeries) vwapLower1sdSeries.applyOptions({{ visible: vwapVisible }});
                if (vwapUpper2sdSeries) vwapUpper2sdSeries.applyOptions({{ visible: vwapVisible }});
                if (vwapLower2sdSeries) vwapLower2sdSeries.applyOptions({{ visible: vwapVisible }});
            }}

            updateMarkers();
        }}

        // Focus on trade from table click
        function focusOnTrade(tradeIdx, entryTime, exitTime) {{
            // If clicking on already selected trade, deselect it
            if (selectedTradeIdx === tradeIdx) {{
                selectedTradeIdx = null;
                document.querySelectorAll('.trades-table tr.selected').forEach(tr => {{
                    tr.classList.remove('selected');
                }});
                updateMarkers();
                return;
            }}

            // Update selected row styling
            document.querySelectorAll('.trades-table tr.selected').forEach(tr => {{
                tr.classList.remove('selected');
            }});
            const selectedRow = document.querySelector(`[data-trade-idx="${{tradeIdx}}"]`);
            if (selectedRow) {{
                selectedRow.classList.add('selected');
            }}

            // Store selected trade
            selectedTradeIdx = tradeIdx;

            // If chart exists, zoom to trade
            if (chart && candleSeries) {{
                const padding = 600; // 10 minutes padding in seconds for more context
                const fromTime = entryTime - padding;
                const toTime = exitTime + padding;

                // Find candles in range (with padding) for Y-axis fitting
                const tradeCandles = ohlcvData.filter(c => c.time >= fromTime && c.time <= toTime);

                if (tradeCandles.length > 0) {{
                    try {{
                        // Set X-axis (time) range
                        chart.timeScale().setVisibleRange({{
                            from: fromTime,
                            to: toTime,
                        }});

                        // Use setTimeout to let the time scale update first,
                        // then reset autoScale to fit Y-axis to visible data
                        setTimeout(() => {{
                            // Toggle autoScale off and on to force recalculation
                            chart.priceScale('right').applyOptions({{ autoScale: false }});
                            chart.priceScale('right').applyOptions({{ autoScale: true }});
                        }}, 50);

                    }} catch (e) {{
                        // Fall back to scrolling to the trade
                        chart.timeScale().scrollToPosition(-10, false);
                    }}
                }}
            }}

            // Update markers with selection highlighting
            updateMarkers();

            // Scroll chart into view
            const chartEl = document.getElementById('chart');
            if (chartEl) chartEl.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }}

        // Initialize charts after DOM is ready
        document.addEventListener('DOMContentLoaded', function() {{
            // Create main chart
            const chartContainer = document.getElementById('chart');
            if (!chartContainer) {{
                console.error('Chart container not found');
                return;
            }}

            chart = LightweightCharts.createChart(chartContainer, {{
                width: chartContainer.clientWidth,
                layout: {{
                    background: {{ type: 'solid', color: '#0a0a0a' }},
                    textColor: '#d1d4dc',
                }},
                grid: {{
                    vertLines: {{ color: '#1a1a1a' }},
                    horzLines: {{ color: '#1a1a1a' }},
                }},
                crosshair: {{
                    mode: LightweightCharts.CrosshairMode.Normal,
                }},
                rightPriceScale: {{
                    borderColor: '#333',
                }},
                timeScale: {{
                    borderColor: '#333',
                    timeVisible: true,
                    secondsVisible: false,
                    minBarSpacing: 0.001,  // Allow extreme zoom out for viewing months/years
                    barSpacing: 6,  // Default bar spacing for reasonable candle width
                    fixLeftEdge: true,
                    fixRightEdge: true,
                }},
            }});

            // Create volume chart
            const volumeContainer = document.getElementById('volume-chart');
            if (!volumeContainer) {{
                console.error('Volume container not found');
                return;
            }}

            volumeChart = LightweightCharts.createChart(volumeContainer, {{
                width: volumeContainer.clientWidth,
                layout: {{
                    background: {{ type: 'solid', color: '#0a0a0a' }},
                    textColor: '#d1d4dc',
                }},
                grid: {{
                    vertLines: {{ color: '#1a1a1a' }},
                    horzLines: {{ visible: false }},
                }},
                rightPriceScale: {{
                    borderColor: '#333',
                    scaleMargins: {{ top: 0.1, bottom: 0 }},
                }},
                timeScale: {{
                    visible: false,
                    minBarSpacing: 0.001,  // Match main chart for perfect alignment
                    barSpacing: 6,
                    fixLeftEdge: true,
                    fixRightEdge: true,
                }},
                handleScroll: false,
                handleScale: false,
            }});

            // Add candlestick series
            candleSeries = chart.addCandlestickSeries({{
                upColor: '#26a69a',
                downColor: '#ef5350',
                borderDownColor: '#ef5350',
                borderUpColor: '#26a69a',
                wickDownColor: '#ef5350',
                wickUpColor: '#26a69a',
                priceLineVisible: false,
                lastValueVisible: false,
            }});
            candleSeries.setData(ohlcvData);

            // Add VWAP line series (daily-anchored) - white main line
            vwapSeries = chart.addLineSeries({{
                color: '#FFFFFF',
                lineWidth: 2,
                lineStyle: 0, // Solid
                priceLineVisible: false,
                lastValueVisible: true,
                crosshairMarkerVisible: true,
                title: 'VWAP',
            }});
            vwapSeries.setData(vwapData);

            // Add VWAP +-1 standard deviation bands (off-white/light gray)
            vwapUpper1sdSeries = chart.addLineSeries({{
                color: 'rgba(200, 200, 200, 0.7)',
                lineWidth: 1,
                lineStyle: 0,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
                title: '+1σ',
            }});
            vwapUpper1sdSeries.setData(vwapUpper1sd);

            vwapLower1sdSeries = chart.addLineSeries({{
                color: 'rgba(200, 200, 200, 0.7)',
                lineWidth: 1,
                lineStyle: 0,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
                title: '-1σ',
            }});
            vwapLower1sdSeries.setData(vwapLower1sd);

            // Add VWAP +-2 standard deviation bands (darker gray)
            vwapUpper2sdSeries = chart.addLineSeries({{
                color: 'rgba(140, 140, 140, 0.5)',
                lineWidth: 1,
                lineStyle: 0,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
                title: '+2σ',
            }});
            vwapUpper2sdSeries.setData(vwapUpper2sd);

            vwapLower2sdSeries = chart.addLineSeries({{
                color: 'rgba(140, 140, 140, 0.5)',
                lineWidth: 1,
                lineStyle: 0,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
                title: '-2σ',
            }});
            vwapLower2sdSeries.setData(vwapLower2sd);

            // Create day boundary vertical lines using a custom primitive
            // This draws actual vertical lines at day boundaries with labels
            window.dayBoundaryMarkers = [];  // Keep empty - we use primitives instead
            // dayBoundaryPrimitive is declared globally for access in switchTimeframe

            if (dailyBoundaries.length > 0) {{
                // Create a custom primitive class for drawing vertical lines
                class DayBoundaryPrimitive {{
                    constructor(boundaries, chart) {{
                        this._boundaries = boundaries;
                        this._chart = chart;
                        this._visible = true;  // Only show for intraday timeframes
                    }}

                    setVisible(visible) {{
                        this._visible = visible;
                        // Trigger redraw
                        this._chart.timeScale().applyOptions({{}});
                    }}

                    isVisible() {{
                        return this._visible;
                    }}

                    updateAllViews() {{
                        // Required method
                    }}

                    paneViews() {{
                        return [new DayBoundaryPaneView(this._boundaries, this._chart, this)];
                    }}
                }}

                class DayBoundaryPaneView {{
                    constructor(boundaries, chart, primitive) {{
                        this._boundaries = boundaries;
                        this._chart = chart;
                        this._primitive = primitive;
                    }}

                    update() {{}}

                    renderer() {{
                        return new DayBoundaryRenderer(this._boundaries, this._chart, this._primitive);
                    }}

                    zOrder() {{
                        return 'bottom';
                    }}
                }}

                class DayBoundaryRenderer {{
                    constructor(boundaries, chart, primitive) {{
                        this._boundaries = boundaries;
                        this._chart = chart;
                        this._primitive = primitive;
                    }}

                    draw(target) {{
                        // Don't draw if not visible (daily or higher timeframes)
                        if (!this._primitive.isVisible()) {{
                            return;
                        }}

                        const timeScale = this._chart.timeScale();
                        const boundaries = this._boundaries;

                        target.useBitmapCoordinateSpace(scope => {{
                            const ctx = scope.context;
                            const height = scope.bitmapSize.height;
                            const width = scope.bitmapSize.width;
                            const pixelRatio = scope.horizontalPixelRatio;

                            ctx.save();
                            ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
                            ctx.lineWidth = 2;
                            ctx.setLineDash([6, 4]);
                            ctx.font = '11px sans-serif';
                            ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
                            ctx.textAlign = 'center';

                            boundaries.forEach(b => {{
                                const logicalX = timeScale.timeToCoordinate(b.time);
                                if (logicalX !== null) {{
                                    // Convert logical to bitmap coordinates
                                    const x = Math.round(logicalX * pixelRatio);
                                    if (x >= 0 && x <= width) {{
                                        // Draw vertical dashed line
                                        ctx.beginPath();
                                        ctx.moveTo(x, 0);
                                        ctx.lineTo(x, height);
                                        ctx.stroke();

                                        // Draw label at bottom
                                        ctx.fillText(b.label, x, height - 8);
                                    }}
                                }}
                            }});

                            ctx.restore();
                        }});
                    }}
                }}

                // Attach the primitive to the candle series
                dayBoundaryPrimitive = new DayBoundaryPrimitive(dailyBoundaries, chart);
                candleSeries.attachPrimitive(dayBoundaryPrimitive);
            }}

            // Custom primitive for trade markers (triangles for entries, circles for exits)
            class TradeMarkerPrimitive {{
                constructor(entries, exits, chart, candleSeries) {{
                    this._entries = entries;
                    this._exits = exits;
                    this._chart = chart;
                    this._candleSeries = candleSeries;
                    this._tradesVisible = true;
                    this._selectedIdx = null;
                }}

                updateState(tradesVisible, selectedIdx) {{
                    this._tradesVisible = tradesVisible;
                    this._selectedIdx = selectedIdx;
                    // Request redraw
                    this._chart.timeScale().applyOptions({{}});
                }}

                updateAllViews() {{}}

                paneViews() {{
                    return [new TradeMarkerPaneView(this)];
                }}
            }}

            class TradeMarkerPaneView {{
                constructor(primitive) {{
                    this._primitive = primitive;
                }}

                update() {{}}

                renderer() {{
                    return new TradeMarkerRenderer(this._primitive);
                }}

                zOrder() {{
                    return 'top';
                }}
            }}

            class TradeMarkerRenderer {{
                constructor(primitive) {{
                    this._primitive = primitive;
                }}

                draw(target) {{
                    const chart = this._primitive._chart;
                    const candleSeries = this._primitive._candleSeries;
                    const entries = this._primitive._entries;
                    const exits = this._primitive._exits;
                    const selectedIdx = this._primitive._selectedIdx;
                    const tradesVisible = this._primitive._tradesVisible;
                    const timeScale = chart.timeScale();

                    target.useBitmapCoordinateSpace(scope => {{
                        const ctx = scope.context;
                        const pixelRatio = scope.horizontalPixelRatio;
                        const vertPixelRatio = scope.verticalPixelRatio;

                        // Calculate zoom-based scale factor and get candle duration
                        let zoomScale = 1.0;
                        let candleDuration = 60; // Default 1m
                        const visibleRange = timeScale.getVisibleRange();
                        if (visibleRange) {{
                            const visibleSeconds = visibleRange.to - visibleRange.from;
                            const eightHours = 8 * 60 * 60;   // 28800 seconds
                            const tenHours = 10 * 60 * 60;    // 36000 seconds
                            if (visibleSeconds <= eightHours) {{
                                zoomScale = 1.0;
                            }} else if (visibleSeconds >= tenHours) {{
                                zoomScale = 0.5;
                            }} else {{
                                // Linear interpolation between 1.0 and 0.5
                                zoomScale = 1.0 - ((visibleSeconds - eightHours) / (tenHours - eightHours)) * 0.5;
                            }}

                            // Determine candle duration based on visible range (match chart timeframe logic)
                            // Thresholds: 1m until 48h visible, then 5m until 8d, etc.
                            const seventyTwoHours = 3 * 24 * 60 * 60;  // 72h = 259200 seconds
                            const eightDays = 8 * 24 * 60 * 60;
                            const thirtyDays = 30 * 24 * 60 * 60;
                            const ninetyDays = 90 * 24 * 60 * 60;
                            const oneYear = 365 * 24 * 60 * 60;

                            if (visibleSeconds < seventyTwoHours) {{
                                candleDuration = 60;          // 1m (less than 72h visible)
                            }} else if (visibleSeconds < eightDays) {{
                                candleDuration = 5 * 60;      // 5m
                            }} else if (visibleSeconds < thirtyDays) {{
                                candleDuration = 15 * 60;     // 15m
                            }} else if (visibleSeconds < ninetyDays) {{
                                candleDuration = 60 * 60;     // 1h
                            }} else if (visibleSeconds < oneYear) {{
                                candleDuration = 24 * 60 * 60; // 1d
                            }} else {{
                                candleDuration = 7 * 24 * 60 * 60; // 1w
                            }}
                        }}

                        // Helper to aggregate markers by time bucket
                        const aggregateByBucket = (markers, isEntry) => {{
                            if (candleDuration <= 60) {{
                                return markers; // 1m - show all markers
                            }}
                            // For higher timeframes, show one marker per candle bucket per direction
                            const seen = new Set();
                            const aggregated = [];
                            for (const m of markers) {{
                                const bucket = Math.floor(m.time / candleDuration) * candleDuration;
                                // For entries: key by bucket + direction (isLong)
                                // For exits: key by bucket + position
                                const direction = isEntry ? (m.isLong ? 'long' : 'short') : m.position;
                                const key = `${{bucket}}_${{direction}}`;
                                if (!seen.has(key)) {{
                                    seen.add(key);
                                    aggregated.push({{ ...m, time: bucket }});
                                }}
                            }}
                            return aggregated;
                        }};

                        // Helper to draw a triangle
                        const drawTriangle = (bx, by, size, pointsUp, color, addBorder) => {{
                            ctx.fillStyle = color;
                            ctx.beginPath();
                            if (pointsUp) {{
                                ctx.moveTo(bx, by - size);
                                ctx.lineTo(bx - size * 0.8, by + size * 0.5);
                                ctx.lineTo(bx + size * 0.8, by + size * 0.5);
                            }} else {{
                                ctx.moveTo(bx, by + size);
                                ctx.lineTo(bx - size * 0.8, by - size * 0.5);
                                ctx.lineTo(bx + size * 0.8, by - size * 0.5);
                            }}
                            ctx.closePath();
                            ctx.fill();
                            if (addBorder) {{
                                ctx.strokeStyle = '#ffffff';
                                ctx.lineWidth = 2;
                                ctx.stroke();
                            }}
                        }};

                        // Draw trade entry triangles
                        if (tradesVisible) {{
                            const aggregatedEntries = aggregateByBucket(entries, true);
                            aggregatedEntries.forEach(m => {{
                                const x = timeScale.timeToCoordinate(m.time);
                                if (x === null) return;

                                const candle = ohlcvData.find(c => c.time === m.time);
                                if (!candle) return;

                                const price = m.isLong ? candle.low : candle.high;
                                const y = candleSeries.priceToCoordinate(price);
                                if (y === null) return;

                                const isSelected = selectedIdx === m.tradeIdx;
                                const size = (isSelected ? 18 : 14) * zoomScale;
                                const offset = (m.isLong ? 28 : -28) * zoomScale;

                                const bx = Math.round(x * pixelRatio);
                                const by = Math.round((y + offset) * vertPixelRatio);

                                drawTriangle(bx, by, size, m.isLong, m.color, isSelected);
                            }});

                            // Draw exit circles (P&L text only on 1m timeframe)
                            const aggregatedExits = aggregateByBucket(exits, false);
                            aggregatedExits.forEach(m => {{
                                const x = timeScale.timeToCoordinate(m.time);
                                if (x === null) return;

                                const candle = ohlcvData.find(c => c.time === m.time);
                                if (!candle) return;

                                const price = m.position === 'aboveBar' ? candle.high : candle.low;
                                const y = candleSeries.priceToCoordinate(price);
                                if (y === null) return;

                                const isSelected = selectedIdx === m.tradeIdx;
                                const radius = (isSelected ? 12 : 10) * zoomScale;
                                const offset = (m.position === 'aboveBar' ? -25 : 25) * zoomScale;

                                const bx = Math.round(x * pixelRatio);
                                const by = Math.round((y + offset) * vertPixelRatio);

                                // Draw circle
                                ctx.fillStyle = m.color;
                                ctx.beginPath();
                                ctx.arc(bx, by, radius, 0, Math.PI * 2);
                                ctx.fill();

                                if (isSelected) {{
                                    ctx.strokeStyle = '#ffffff';
                                    ctx.lineWidth = 2;
                                    ctx.stroke();
                                }}

                                // Only draw P&L text on 1m timeframe
                                if (candleDuration <= 60) {{
                                    ctx.fillStyle = m.color;
                                    ctx.font = 'bold 14px sans-serif';
                                    ctx.textAlign = 'center';
                                    const textOffset = m.position === 'aboveBar' ? -19 : 21;
                                    ctx.fillText(m.text, bx, by + textOffset);
                                }}
                            }});
                        }}
                    }});
                }}
            }}

            // Create and attach trade marker primitive (trades only, signals use built-in markers)
            window.tradeMarkerPrimitive = new TradeMarkerPrimitive(tradeEntries, tradeExits, chart, candleSeries);
            candleSeries.attachPrimitive(window.tradeMarkerPrimitive);

            // Add volume series
            volumeSeries = volumeChart.addHistogramSeries({{
                priceFormat: {{ type: 'volume' }},
                priceScaleId: 'right',
                priceLineVisible: false,
                lastValueVisible: false,
            }});
            volumeSeries.setData(volumeData);

            // Get data bounds (used for initial view)
            const dataMinTime = ohlcvData1m.length > 0 ? ohlcvData1m[0].time : 0;
            const dataMaxTime = ohlcvData1m.length > 0 ? ohlcvData1m[ohlcvData1m.length - 1].time : 0;

            // Sync time scales and handle timeframe switching
            // Note: fixLeftEdge/fixRightEdge on timeScale options handle zoom constraints
            chart.timeScale().subscribeVisibleTimeRangeChange((range) => {{
                if (!range || !volumeChart) return;

                // Sync volume chart with main chart
                try {{
                    volumeChart.timeScale().setVisibleRange(range);
                }} catch (e) {{
                    // Ignore sync errors
                }}

                // Check if we need to switch timeframes based on visible range
                const visibleSeconds = range.to - range.from;
                const newTf = getTimeframeForRange(visibleSeconds);
                if (newTf !== currentTimeframe) {{
                    // Store current range before switching
                    const currentRange = {{ from: range.from, to: range.to }};
                    switchTimeframe(newTf);
                    // Restore the range after data switch
                    try {{
                        chart.timeScale().setVisibleRange(currentRange);
                        volumeChart.timeScale().setVisibleRange(currentRange);
                    }} catch (e) {{
                        // Ignore range restoration errors
                    }}
                }}
            }});

            // Set initial disabled state on legend items
            ['1m', '5m', '15m', 'breach'].forEach(s => {{
                const item = document.querySelector(`[data-series="${{s}}"]`);
                if (item) item.classList.add('disabled');
            }});

            // Initial marker setup
            updateMarkers();

            // Function to resize charts based on container
            function resizeCharts() {{
                const chartWrapper = document.querySelector('.chart-wrapper');
                const volumeHeight = 60;
                const mainChartHeight = chartWrapper.clientHeight - volumeHeight;

                chart.applyOptions({{
                    width: chartContainer.clientWidth,
                    height: Math.max(200, mainChartHeight),
                }});
                volumeChart.applyOptions({{
                    width: volumeContainer.clientWidth,
                    height: volumeHeight,
                }});
            }}

            // Initial resize and fit to exact data bounds (no padding)
            resizeCharts();
            chart.timeScale().setVisibleRange({{ from: dataMinTime, to: dataMaxTime }});
            volumeChart.timeScale().setVisibleRange({{ from: dataMinTime, to: dataMaxTime }});

            // Handle resize
            window.addEventListener('resize', resizeCharts);

            // Sync crosshairs between all charts
            function syncCrosshair(sourceChart, targetCharts, param) {{
                if (param.time) {{
                    targetCharts.forEach(targetChart => {{
                        if (targetChart) {{
                            targetChart.setCrosshairPosition(param.seriesData?.get(candleSeries)?.close || 0, param.time, candleSeries);
                        }}
                    }});
                }} else {{
                    targetCharts.forEach(targetChart => {{
                        if (targetChart) {{
                            targetChart.clearCrosshairPosition();
                        }}
                    }});
                }}
            }}

            chart.subscribeCrosshairMove((param) => {{
                syncCrosshair(chart, [volumeChart], param);
            }});

            // Handle chart clicks for bidirectional selection
            chart.subscribeClick((param) => {{
                if (!param.time || !seriesState.trades) return;

                const clickTime = param.time;

                // Find trade that contains this time
                for (let i = 0; i < tradeRanges.length; i++) {{
                    const trade = tradeRanges[i];
                    // Check if click is near entry or exit time (within 60 seconds)
                    const nearEntry = Math.abs(clickTime - trade.entry) < 60;
                    const nearExit = Math.abs(clickTime - trade.exit) < 60;

                    if (nearEntry || nearExit) {{
                        selectedTradeIdx = i;

                        // Switch to Trades tab
                        switchTab('trades');

                        // Update table selection
                        document.querySelectorAll('.trades-table tr.selected').forEach(tr => {{
                            tr.classList.remove('selected');
                        }});
                        const selectedRow = document.querySelector(`[data-trade-idx="${{i}}"]`);
                        if (selectedRow) {{
                            selectedRow.classList.add('selected');
                            selectedRow.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                        }}

                        updateMarkers();
                        return;
                    }}
                }}

                // Click wasn't near any trade marker - deselect
                selectedTradeIdx = null;
                document.querySelectorAll('.trades-table tr.selected').forEach(tr => {{
                    tr.classList.remove('selected');
                }});
                updateMarkers();
            }});

            // Clear selection when double-clicking on chart
            chartContainer.addEventListener('dblclick', (e) => {{
                selectedTradeIdx = null;
                document.querySelectorAll('.trades-table tr.selected').forEach(tr => {{
                    tr.classList.remove('selected');
                }});
                updateMarkers();
            }});

            // P/L chart is now created inline and synced with main chart
        }}); // End DOMContentLoaded
    </script>
</body>
</html>
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)

    return output_path
