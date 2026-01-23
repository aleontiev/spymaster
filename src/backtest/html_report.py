"""
HTML Report Generator for Multi-Scale Backtest.

Generates interactive TradingView-style charts with:
- OHLCV candlestick data with volume subchart
- Trade entries/exits with win/loss coloring
- Individual model signals (1m, 5m, 15m) as toggleable series
- Daily boundary lines with date labels
- Bidirectional trade selection (table <-> chart)
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


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


def generate_html_report(
    ohlcv_df: pd.DataFrame,
    trades: List[TradeMarker],
    signals: Dict[str, List[SignalMarker]],  # {"1m": [...], "5m": [...], "15m": [...]}
    combination_name: str,
    output_path: Path,
    title: str = "Multi-Scale Backtest Report",
    command: Optional[str] = None,
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

    Returns:
        Path to the generated HTML file
    """
    # Convert OHLCV to JSON format for Lightweight Charts
    ohlcv_data = []
    volume_data = []
    for idx, row in ohlcv_df.iterrows():
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
        volume_data.append({
            "time": ts,
            "value": float(row.get("volume", 0)),
            "color": "rgba(38, 166, 154, 0.5)" if is_up else "rgba(239, 83, 80, 0.5)",
        })

    # Calculate daily boundaries for vertical lines
    daily_boundaries = []
    current_date = None
    for idx, row in ohlcv_df.iterrows():
        ts = int(idx.timestamp())
        dt = idx.tz_convert("America/New_York") if idx.tzinfo else idx
        date_str = dt.strftime("%b %d")
        if current_date != dt.date():
            if current_date is not None:
                daily_boundaries.append({
                    "time": ts,
                    "label": date_str,
                })
            current_date = dt.date()

    # Convert signals to JSON
    signals_json = {}
    for model, signal_list in signals.items():
        signals_json[model] = [
            {
                "time": s.timestamp,
                "position": "aboveBar" if s.signal_type == "short" else "belowBar",
                "color": "#ef5350" if s.signal_type == "short" else "#26a69a",
                "shape": "arrowDown" if s.signal_type == "short" else "arrowUp",
                "text": "",  # No text label - just arrows
                "size": 1,
            }
            for s in signal_list
        ]

    # Convert trades to entry/exit markers - simplified labels
    trade_entries = []
    trade_exits = []
    for i, t in enumerate(trades):
        # For runner trades, use parent_entry_time for entry marker
        entry_ts = t.parent_entry_time if t.is_runner and t.parent_entry_time else t.entry_time

        # Entry marker - no text label
        entry_color = "#26a69a" if t.is_long else "#ef5350"
        trade_entries.append({
            "time": entry_ts,
            "position": "belowBar" if t.is_long else "aboveBar",
            "color": entry_color,
            "shape": "arrowUp" if t.is_long else "arrowDown",
            "text": "",  # No label for entries
            "size": 2,
            "tradeIdx": i,
        })

        # Exit marker - just show P&L %
        exit_color = "#26a69a" if t.is_win else "#ef5350"
        exit_label = f"{t.pnl_pct:+.0f}%"
        trade_exits.append({
            "time": t.exit_time,
            "position": "aboveBar" if t.is_long else "belowBar",
            "color": exit_color,
            "shape": "circle",
            "text": exit_label,
            "size": 2,
            "tradeIdx": i,
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

        trades_table_rows.append(f"""
            <tr class="{result_class}" data-trade-idx="{i}" data-entry-time="{t.entry_time}" data-exit-time="{t.exit_time}" onclick="focusOnTrade({i}, {t.entry_time}, {t.exit_time})" style="cursor: pointer;">
                <td>{i+1}</td>
                <td>{entry_dt}</td>
                <td>{duration_str}</td>
                <td>{contract}</td>
                <td>{t.num_contracts}</td>
                <td>${t.entry_option_price:.2f}</td>
                <td>${t.exit_option_price:.2f}</td>
                <td class="pnl">{t.pnl_pct:+.2f}%</td>
                <td class="pnl">${t.pnl_dollars:+,.2f}</td>
                <td>${t.capital_after:,.0f}</td>
                <td>{t.exit_reason}</td>
            </tr>
        """)

    trades_table_html = "\n".join(trades_table_rows)

    # Calculate summary stats
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.is_win)
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    total_pnl = sum(t.pnl_dollars for t in trades)

    # Calculate Max Drawdown %, Max Gain %, Total %
    initial_capital = trades[0].capital_after - trades[0].pnl_dollars if trades else 100000
    running_capital = initial_capital
    peak_capital = initial_capital
    max_drawdown_pct = 0
    max_gain_pct = 0

    for t in trades:
        running_capital = t.capital_after
        if running_capital > peak_capital:
            peak_capital = running_capital
        drawdown = (peak_capital - running_capital) / peak_capital * 100
        if drawdown > max_drawdown_pct:
            max_drawdown_pct = drawdown
        gain = (running_capital - initial_capital) / initial_capital * 100
        if gain > max_gain_pct:
            max_gain_pct = gain

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
    for model in ["1m", "5m", "15m"]:
        if model in signal_stats:
            s = signal_stats[model]
            signal_stats_rows += f"""
                <tr>
                    <td><strong>{model.upper()}</strong></td>
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

    # Calculate monthly P&L
    monthly_stats = {}
    for t in trades:
        month_key = datetime.fromtimestamp(t.entry_time).strftime("%Y-%m")
        if month_key not in monthly_stats:
            monthly_stats[month_key] = {"trades": 0, "pnl": 0.0, "wins": 0, "balance": 0.0}
        monthly_stats[month_key]["trades"] += 1
        monthly_stats[month_key]["pnl"] += t.pnl_dollars
        monthly_stats[month_key]["balance"] = t.capital_after
        if t.is_win:
            monthly_stats[month_key]["wins"] += 1

    monthly_rows = ""
    for month in sorted(monthly_stats.keys()):
        m = monthly_stats[month]
        win_rate_m = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
        pnl_class = "positive" if m["pnl"] >= 0 else "negative"
        monthly_rows += f"""
            <tr>
                <td><strong>{month}</strong></td>
                <td>{m['trades']}</td>
                <td>{win_rate_m:.1f}%</td>
                <td class="{pnl_class}">${m['pnl']:+,.0f}</td>
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
            weekly_stats[week_key] = {"trades": 0, "pnl": 0.0, "wins": 0, "balance": 0.0}
        weekly_stats[week_key]["trades"] += 1
        weekly_stats[week_key]["pnl"] += t.pnl_dollars
        weekly_stats[week_key]["balance"] = t.capital_after
        if t.is_win:
            weekly_stats[week_key]["wins"] += 1

    weekly_rows = ""
    for week in sorted(weekly_stats.keys()):
        w = weekly_stats[week]
        win_rate_w = w["wins"] / w["trades"] * 100 if w["trades"] > 0 else 0
        pnl_class = "positive" if w["pnl"] >= 0 else "negative"
        weekly_rows += f"""
            <tr>
                <td><strong>{week}</strong></td>
                <td>{w['trades']}</td>
                <td>{win_rate_w:.1f}%</td>
                <td class="{pnl_class}">${w['pnl']:+,.0f}</td>
                <td>${w['balance']:,.0f}</td>
            </tr>
        """

    # Extract date range info from OHLCV data
    if len(ohlcv_df) > 0:
        first_ts = ohlcv_df.index[0]
        last_ts = ohlcv_df.index[-1]
        start_date_str = first_ts.strftime("%Y-%m-%d")
        end_date_str = last_ts.strftime("%Y-%m-%d")
    else:
        start_date_str = "N/A"
        end_date_str = "N/A"

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
        .stat-label {{
            font-size: 11px;
            color: #888;
            margin-top: 2px;
        }}
        /* Section 3: Chart (~45% of remaining) */
        .chart-section {{
            flex: 45;
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
            flex: 55;
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
            display: block;
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
        <span class="header-item">Models: <strong>{combination_name}</strong></span>
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
            <div class="stat-value positive">{max_gain_pct:+.1f}%</div>
            <div class="stat-label">Max Gain</div>
        </div>
        <div class="stat">
            <div class="stat-value negative">-{max_drawdown_pct:.1f}%</div>
            <div class="stat-label">Max Drawdown</div>
        </div>
    </div>

    <!-- Section 3: Chart -->
    <div class="chart-section">
        <div class="legend">
            <div class="legend-item" data-series="trades" onclick="toggleSeries('trades')">
                <div class="legend-color" style="background: linear-gradient(135deg, #26a69a 50%, #ef5350 50%);"></div>
                <span>Trades ({total_trades})</span>
            </div>
            <div class="legend-item" data-series="1m" onclick="toggleSeries('1m')">
                <div class="legend-color" style="background: #ffeb3b;"></div>
                <span>1M ({signal_counts.get('1m', 0)})</span>
            </div>
            <div class="legend-item" data-series="5m" onclick="toggleSeries('5m')">
                <div class="legend-color" style="background: #ff9800;"></div>
                <span>5M ({signal_counts.get('5m', 0)})</span>
            </div>
            <div class="legend-item" data-series="15m" onclick="toggleSeries('15m')">
                <div class="legend-color" style="background: #9c27b0;"></div>
                <span>15M ({signal_counts.get('15m', 0)})</span>
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
            <button class="tab" onclick="switchTab('monthly')">By Month</button>
            <button class="tab" onclick="switchTab('weekly')">By Week</button>
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
                        <th>Exit Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {trades_table_html}
                </tbody>
            </table>
        </div>
        <div id="monthly-tab" class="tab-content">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Month</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>P&L</th>
                        <th>Balance</th>
                    </tr>
                </thead>
                <tbody>
                    {monthly_rows}
                </tbody>
            </table>
        </div>
        <div id="weekly-tab" class="tab-content">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>Week Start</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>P&L</th>
                        <th>Balance</th>
                    </tr>
                </thead>
                <tbody>
                    {weekly_rows}
                </tbody>
            </table>
        </div>
        <div id="details-tab" class="tab-content">
            <div class="details-section">
                {f'<div class="details-block"><h3>Command Used</h3><pre>{command}</pre></div>' if command else ''}
                <div class="details-block">
                    <h3>Summary Statistics</h3>
                    <div class="details-grid">
                        <div class="detail-item">
                            <div class="detail-label">Initial Capital</div>
                            <div class="detail-value">${initial_capital:,.0f}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Final Capital</div>
                            <div class="detail-value">${running_capital:,.0f}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Total P&L</div>
                            <div class="detail-value" style="color: {'#26a69a' if total_pnl >= 0 else '#ef5350'}">${total_pnl:+,.0f}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Win Count</div>
                            <div class="detail-value">{wins} / {total_trades}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Trading Days</div>
                            <div class="detail-value">{int(trading_days)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Trades per Day</div>
                            <div class="detail-value">{total_trades/trading_days:.1f}</div>
                        </div>
                    </div>
                </div>
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
        // Data
        const ohlcvData = {json.dumps(ohlcv_data)};
        const volumeData = {json.dumps(volume_data)};
        const dailyBoundaries = {json.dumps(daily_boundaries)};
        const tradeEntries = {json.dumps(trade_entries)};
        const tradeExits = {json.dumps(trade_exits)};
        const tradeRanges = {json.dumps(trade_ranges)};
        const signals1m = {json.dumps(signals_json.get('1m', []))};
        const signals5m = {json.dumps(signals_json.get('5m', []))};
        const signals15m = {json.dumps(signals_json.get('15m', []))};

        // Track series visibility - MUST be defined before chart creation
        const seriesState = {{
            trades: true,
            '1m': true,
            '5m': true,
            '15m': true,
        }};

        // Track selected trade - MUST be defined before chart creation
        let selectedTradeIdx = null;

        // Tab switching
        function switchTab(tabName) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`[onclick="switchTab('${{tabName}}')"]`).classList.add('active');
            document.getElementById(`${{tabName}}-tab`).classList.add('active');
        }}

        // Toggle series visibility - defined early for onclick handlers
        function toggleSeries(seriesName) {{
            seriesState[seriesName] = !seriesState[seriesName];
            const legendItem = document.querySelector(`[data-series="${{seriesName}}"]`);
            if (legendItem) {{
                legendItem.classList.toggle('disabled', !seriesState[seriesName]);
            }}
            if (typeof updateMarkers === 'function') updateMarkers();
        }}

        // Focus on trade - defined early for onclick handlers
        // Will be overwritten with full implementation after chart is created
        let focusOnTradeImpl = null;
        function focusOnTrade(tradeIdx, entryTime, exitTime) {{
            if (focusOnTradeImpl) {{
                focusOnTradeImpl(tradeIdx, entryTime, exitTime);
            }} else {{
                // Fallback: just update selection
                selectedTradeIdx = tradeIdx;
                document.querySelectorAll('.trades-table tr.selected').forEach(tr => tr.classList.remove('selected'));
                const row = document.querySelector(`[data-trade-idx="${{tradeIdx}}"]`);
                if (row) row.classList.add('selected');
            }}
        }}

        // Create main chart
        const chartContainer = document.getElementById('chart');
        const chart = LightweightCharts.createChart(chartContainer, {{
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
            }},
        }});

        // Create volume chart
        const volumeContainer = document.getElementById('volume-chart');
        const volumeChart = LightweightCharts.createChart(volumeContainer, {{
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
            }},
            handleScroll: false,
            handleScale: false,
        }});

        // Sync time scales
        chart.timeScale().subscribeVisibleTimeRangeChange((range) => {{
            if (range) {{
                volumeChart.timeScale().setVisibleRange(range);
            }}
        }});

        // Add candlestick series
        const candleSeries = chart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderDownColor: '#ef5350',
            borderUpColor: '#26a69a',
            wickDownColor: '#ef5350',
            wickUpColor: '#26a69a',
        }});
        candleSeries.setData(ohlcvData);

        // Create day boundary markers (will be added to markers in updateMarkers)
        const dayBoundaryMarkers = dailyBoundaries.map(b => ({{
            time: b.time,
            position: 'aboveBar',
            color: 'rgba(100, 100, 100, 0.8)',
            shape: 'square',
            text: b.label,
            size: 0,
        }}));

        // Add volume series
        const volumeSeries = volumeChart.addHistogramSeries({{
            priceFormat: {{ type: 'volume' }},
            priceScaleId: 'right',
        }});
        volumeSeries.setData(volumeData);

        // Function to update markers based on visibility
        function updateMarkers() {{
            let allMarkers = [];

            // Always show day boundary markers
            allMarkers = allMarkers.concat(dayBoundaryMarkers);

            if (seriesState.trades) {{
                const entries = tradeEntries.map((m, idx) => {{
                    const isSelected = selectedTradeIdx === m.tradeIdx;
                    return {{
                        ...m,
                        size: isSelected ? 3 : 2,
                        // Add white border effect for selection by using a brighter shade
                        color: isSelected ? '#ffffff' : m.color,
                    }};
                }});
                const exits = tradeExits.map((m, idx) => {{
                    const isSelected = selectedTradeIdx === m.tradeIdx;
                    return {{
                        ...m,
                        size: isSelected ? 3 : 2,
                        color: isSelected ? '#ffffff' : m.color,
                    }};
                }});
                allMarkers = allMarkers.concat(entries, exits);
            }}

            if (seriesState['1m']) {{
                const markers1m = signals1m.map(m => ({{
                    ...m,
                    color: '#ffeb3b',
                    size: 1,
                }}));
                allMarkers = allMarkers.concat(markers1m);
            }}

            if (seriesState['5m']) {{
                const markers5m = signals5m.map(m => ({{
                    ...m,
                    color: '#ff9800',
                    size: 1,
                }}));
                allMarkers = allMarkers.concat(markers5m);
            }}

            if (seriesState['15m']) {{
                const markers15m = signals15m.map(m => ({{
                    ...m,
                    color: '#9c27b0',
                    size: 1,
                }}));
                allMarkers = allMarkers.concat(markers15m);
            }}

            // Sort by time
            allMarkers.sort((a, b) => a.time - b.time);

            candleSeries.setMarkers(allMarkers);

            // Update candle colors for selection highlighting
            if (selectedTradeIdx !== null) {{
                const trade = tradeRanges[selectedTradeIdx];
                const highlightedData = ohlcvData.map(candle => {{
                    if (candle.time >= trade.entry && candle.time <= trade.exit) {{
                        return {{
                            ...candle,
                            borderColor: '#26a69a',
                            wickColor: '#26a69a',
                        }};
                    }}
                    return candle;
                }});
                candleSeries.setData(highlightedData);
            }} else {{
                candleSeries.setData(ohlcvData);
            }}
        }}

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

        // Initial resize and fit
        resizeCharts();
        chart.timeScale().fitContent();
        volumeChart.timeScale().fitContent();

        // Handle resize
        window.addEventListener('resize', resizeCharts);

        // Set the full implementation of focusOnTrade now that chart exists
        focusOnTradeImpl = function(tradeIdx, entryTime, exitTime) {{
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

            // Calculate time range with padding
            const padding = 300; // 5 minutes in seconds
            const fromTime = entryTime - padding;
            const toTime = exitTime + padding;

            // Set visible range
            chart.timeScale().setVisibleRange({{
                from: fromTime,
                to: toTime,
            }});

            // Update markers with selection highlighting
            updateMarkers();

            // Scroll chart into view
            document.getElementById('chart').scrollIntoView({{ behavior: 'smooth', block: 'center' }});
        }};

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

        // Clear selection when clicking elsewhere on chart
        chartContainer.addEventListener('dblclick', (e) => {{
            selectedTradeIdx = null;
            document.querySelectorAll('.trades-table tr.selected').forEach(tr => {{
                tr.classList.remove('selected');
            }});
            updateMarkers();
        }});
    </script>
</body>
</html>
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content)

    return output_path
